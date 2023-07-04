import torch
from preprocess import CountryTokenizer,NameDataset,collate_fn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
from rnnclassifier import RNNClassifier
from torch import nn
from utils import time_since
from utils import l2_regularization

EPOCH_NUM = 50
BATCH_SIZE = 8
N_CHARS = 128
HIDDEN_SIZE = 100
N_LAYER = 2
N_EPOCHS = 100

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# TRAIN
train_file_path = "names_train.csv"
tokenizer = CountryTokenizer(train_file_path)
train_data = NameDataset(train_file_path)
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)

# TEST
test_file_path = "names_test.csv"
test_data = NameDataset(train_file_path)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)


# 模型
model = RNNClassifier(N_CHARS, HIDDEN_SIZE, tokenizer.get_country_size(), N_LAYER).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

start = time.time()
accuracy_list = []
print(f"Training for {EPOCH_NUM} epochs...")

for epoch in range(1, EPOCH_NUM + 1):
    # Train
    model.train()
    total_step = len(train_loader)
    for i, (tensor_seq, len_seq, country_seq) in enumerate(train_loader, 1):
        # len_seq必须在cpu上，不要转换（因为pack_padded_sequence的要求）
        tensor_seq, country_seq = tensor_seq.to(device), country_seq.to(device)

        output = model(tensor_seq, len_seq)

        #add L2_regularization
        l2_loss=l2_regularization(model,0.4)
        loss = criterion(output, country_seq)+l2_loss


        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 10 == 0:
            print(
                f'Train... [time_cost {time_since(start)}] \t [Epoch {epoch}/{N_EPOCHS}, Step {i}/{total_step}] \t [loss={loss.item()}]')

    # Eval
    model.eval()
    correct, total = 0, 0
    for tensor_seq, len_seq, country_seq in test_loader:
        tensor_seq, country_seq = tensor_seq.to(device), country_seq.to(device)

        with torch.no_grad():
            output = model(tensor_seq, len_seq)

        output = output.argmax(dim=1)
        correct += (output == country_seq).sum().item()
        total += len(country_seq)

    print(
        f'Eval...  [time_cost {time_since(start)}] \t [Epoch {epoch}/{N_EPOCHS}] \t [accuracy = {(100 * correct / total)}%]')

    accuracy_list.append(correct / total)

plt.figure(figsize=(12.8, 7.2))
plt.plot(range(0, EPOCH_NUM), accuracy_list, label="train")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.show()


my_data = [
    ('Li', 'Chinese'), ('Victor', 'French'), ('wang', 'Chinese'),('Laurent',"French")
]

tensor_seq, len_seq, country_seq = collate_fn(my_data)

with torch.no_grad():
    tensor_seq, country_seq = tensor_seq.to(device), country_seq.to(device)

    output = model(tensor_seq, len_seq)
    output = output.argmax(dim=1)

    print("Names   ", ["".join([chr(a) for a in n_seq if a > 0]) for n_seq in tensor_seq.tolist()])
    print("Predict ", [tokenizer.decode(i) for i in output.tolist()])
    print("Real    ", [tokenizer.decode(i) for i in country_seq.tolist()])
