RED = \033[0;31m
GREEN = \033[0;32m
NC = \033[0m

deps:
	@echo "$(GREEN)[*]Installing deps...$(NC)"
	pip3 install -r requirements.txt
	@echo "$(GREEN)[+]Installing deps finished!$(NC)"

train:
	@echo "$(GREEN)[*]Running train script...$(NC)"
	@chmod +x ./train.sh
	./train_llama.sh
	@echo "$(GREEN)[+]Running train script finished!$(NC)"

infer:
	@echo "$(GREEN)[*]Running inference script...$(NC)"
	@chmod +x ./inference.sh
	./inference.sh
	@echo "$(GREEN)[+]Running inference script finished!$(NC)"

infer2:
	@echo "$(GREEN)[*]Running inference2 script...$(NC)"
	@chmod +x ./inference2.sh
	./inference2.sh
	@echo "$(GREEN)[+]Running inference2 script...$(NC)"

inferOther:
	@echo "$(GREEN)[*]Running inferenceOther script...$(NC)"
	@chmod +x ./inference_other.sh
	./inference_other.sh
	@echo "$(GREEN)[+]Running inferenceOther script...$(NC)"

all: deps train infer infer2 inferOther
	@echo "$(GREEN)[+]All done!$(NC)"

.PHONY: all deps train infer infer2