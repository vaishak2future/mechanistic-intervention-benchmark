ARG VLLM_IMAGE=vllm/vllm-openai:v0.11.0
FROM ${VLLM_IMAGE}
RUN pip install --no-cache-dir peft==0.17.1 accelerate==1.10.1
COPY eval.py TASK.txt tasks.json /task/
RUN mkdir /workspace && chown 1000:1000 /workspace
ENV HOME=/tmp HF_HOME=/tmp/huggingface HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1
USER 1000:1000
WORKDIR /workspace
ENTRYPOINT []
CMD ["sleep", "infinity"]
