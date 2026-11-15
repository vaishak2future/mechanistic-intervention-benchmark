FROM vllm/vllm-openai:v0.11.0

USER root
ENTRYPOINT []
RUN ln -sf /usr/bin/python3 /usr/local/bin/python
RUN python -m pip install peft==0.17.1 accelerate==1.10.1
RUN mkdir -p /base /task /workspace /prepared /cache

COPY download_model.py /task/download_model.py
RUN python /task/download_model.py robbiemu/MobileLLM-R1-950M-mlx-fp16 --revision cdd21efc69cc89fa9608cfbad863f9eebb9dc9fb

ENV HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1
ENV VLLM_CACHE_ROOT=/cache/vllm TORCHINDUCTOR_CACHE_DIR=/cache/inductor TRITON_CACHE_DIR=/cache/triton
ENV PYTHONPATH=/pilot VLLM_ALLOW_INSECURE_SERIALIZATION=1
RUN mkdir -p /pilot
COPY eval.py patch.py prompts.py evaluate.py metrics.py scoring.py dev.json TASK.txt ready.py /pilot/
WORKDIR /workspace
