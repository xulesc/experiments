
download_llama2:
	curl -C - -L https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q8_0.bin \
		--output llama-2-7b-chat.ggmlv3.q8_0.bin --http1.1

create_venv:
	python -m venv .venv

activate_venv:
	source .venv/bin/activate

install_packages:
	pip install llama-cpp-python
