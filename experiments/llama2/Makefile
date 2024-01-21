
deps:
	@echo "Installing deps..."
	pip3 install -r requirements.txt

train:
	@echo "Running train script..."
	@chmod +x ./train_llama.sh
	./train_llama.sh

infer:
	@echo "Running inference script..."
	@chmod +x ./inference.sh
	./inference.sh

infer2:
	@echo "Running inference2 script..."
	@chmod +x ./inference2.sh
	./inference2.sh

all: deps train infer infer2
