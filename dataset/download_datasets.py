from datasets import load_dataset
import os

# Download TriviaQA dataset
print("Downloading TriviaQA dataset...")
trivia_qa = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")
trivia_qa.save_to_disk("./trivia_qa")
print("TriviaQA dataset saved to ./trivia_qa")

# Download SQuAD dataset
print("Downloading SQuAD dataset...")
squad = load_dataset("rajpurkar/squad")
squad.save_to_disk("./squad")
print("SQuAD dataset saved to ./squad")

print("Done! Both datasets have been downloaded and saved.")