from components.base_component import PipelineComponent
import os
import time
import torch
from utils.config_loader import config
from utils.storage_manager import StorageManager
from utils.runtime_config_loader import RuntimeConfig
from components.asr.openai.whisper import Whisper as OA_Whisper
from components.asr.diarization.pyannote_diarizer import PyannoteDiarizer
from components.asr.openvino.whisper import Whisper as OV_Whisper
from components.asr.funasr.paraformer import Paraformer
import logging
logger = logging.getLogger(__name__)

ENABLE_DIARIZATION = config.models.asr.diarization
DELETE_CHUNK_AFTER_USE =  config.pipeline.delete_chunks_after_use
threads_limit = config.models.asr.threads_limit
THREADS_LIMIT = threads_limit if threads_limit and threads_limit > 0 else None
class ASRComponent(PipelineComponent):

    _model = None
    _config = None

    def __init__(self, session_id, provider="openai", model_name="whisper-small", device="CPU", temperature=0.0):

        self.session_id = session_id
        self.temperature = temperature
        self.provider = provider
        self.model_name = model_name
        self.speaker_text_len = {}   # accumulate across chunks
        self.threads_limit = THREADS_LIMIT
        self.enable_diarization = ENABLE_DIARIZATION
        provider, model_name = provider.lower(), model_name.lower()
        model_config_key = (provider, model_name, device)

        # Reload only if config changed
        if ASRComponent._model is None or ASRComponent._config != model_config_key:
            if provider == "openai" and "whisper" in model_name:
                ASRComponent._model = OA_Whisper(model_name, device, None)
            elif provider == "openvino" and "whisper" in model_name:
                ASRComponent._model = OV_Whisper(model_name, device, None,self.threads_limit)
            elif provider == "funasr" and "paraformer" in model_name:
                ASRComponent._model = Paraformer(model_name, device.lower(), None)
            else:
                raise ValueError(f"Unsupported ASR provider/model: {provider}/{model_name}")
            ASRComponent._config = model_config_key

        self.asr = ASRComponent._model
        
        self.pyannote_diarizer = None
        if self.enable_diarization:
            self.pyannote_diarizer = PyannoteDiarizer(
                hf_token=config.models.asr.hf_token
            )

    def process(self, input_generator):

        project_config = RuntimeConfig.get_section("Project")
        project_path = os.path.join(project_config.get("location"), project_config.get("name"), self.session_id)
        StorageManager.save(os.path.join(project_path, "transcription.txt"), "", append=False)

        start_time = time.perf_counter()
        default_torch_threads = None
        try: 
            if self.provider in ["openai", "funasr"] and self.threads_limit and self.threads_limit > 0:
                default_torch_threads = torch.get_num_threads()
                torch.set_num_threads(self.threads_limit)

            for chunk_data in input_generator:
                chunk_path = chunk_data["chunk_path"]
                transcription = self.asr.transcribe(chunk_path, temperature=self.temperature)

                ui_segments = []        # <-- structured for frontend
                transcribed_text = ""   # <-- flat text for file saving

                if self.enable_diarization and transcription["segments"]:
                    # Speaker turns from Pyannote
                    speaker_turns = self.pyannote_diarizer.diarize(chunk_path)
                    logger.debug(f"[DIARIZER] Produced {len(speaker_turns)} speaker turns")

                    timeline_segments = []     # frontend output
                    speaker_lines = []         # text file output

                    for sent in transcription["segments"]:

                        if not sent["text"].strip():
                            continue
                        mid = (sent["start"] + sent["end"]) / 2.0

                        speaker = "UNKNOWN"
                        for turn in speaker_turns:
                            if turn["start"] <= mid <= turn["end"]:
                                speaker = turn["speaker"]
                                break

                        text = sent["text"].strip()
                        start = float(sent["start"])
                        end   = float(sent["end"])

                        # ---- frontend timeline entry ----
                        timeline_segments.append({
                            "speaker": speaker,
                            "text": text,
                            "start": start,
                            "end": end
                        })

                        # ---- Accumulate text length per speaker ----
                        if speaker != "UNKNOWN":   # ignore unknown
                            self.speaker_text_len[speaker] = (
                                self.speaker_text_len.get(speaker, 0) + len(text)
                            )

                        # ---- file line ----
                        speaker_lines.append(f"{speaker}: {text}")

                    transcribed_text = "\n".join(speaker_lines) + "\n"

                    # Save speaker transcript file
                    StorageManager.save_async(
                        os.path.join(project_path, "speaker_transcription.txt"),
                        "\n".join(speaker_lines) + "\n",
                        append=True
                    )

                else:
                    transcribed_text = transcription["text"]
                    timeline_segments = [{
                        "speaker": None,
                        "text": transcribed_text,
                        "start": 0.0,
                        "end": 0.0
                    }]

                # cleanup chunk
                if os.path.exists(chunk_path) and DELETE_CHUNK_AFTER_USE:
                    os.remove(chunk_path)

                # save flat transcript
                StorageManager.save_async(
                    os.path.join(project_path, "transcription.txt"),
                    transcribed_text,
                    append=True
                )

                # yield BOTH formats
                yield {
                    **chunk_data,
                    "text": transcribed_text,        # for legacy usage
                    "segments": ui_segments          # <-- frontend uses this
                }

            teacher_speaker = None
            if self.speaker_text_len:
                teacher_speaker = max(self.speaker_text_len, key=self.speaker_text_len.get)

            yield {
                "event": "final",
                "teacher_speaker": teacher_speaker,
                "speaker_text_stats": self.speaker_text_len
            }


        finally:
            if default_torch_threads is not None:
                torch.set_num_threads(default_torch_threads)
                
            end_time = time.perf_counter()
            transcription_time = end_time - start_time

            # Save the transcription time in the metrics CSV file
            StorageManager.update_csv(
                path=os.path.join(project_path, "performance_metrics.csv"),
                new_data={
                    "configuration.asr_model": f"{self.provider}/{self.model_name}",
                    "performance.transcription_time": round(transcription_time, 4)
                }
            )

            logger.info(f"Transcription Complete: {self.session_id}")

            
