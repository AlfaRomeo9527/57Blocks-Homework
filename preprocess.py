import csv
import torch

class CountryTokenizer:

    def __init__(self, file_path):
        # 生成语言对应的编码词典
        self.country2idx_dict = dict()
        idx = 0
        with open(file_path, "rt") as f:
            reader = csv.reader(f)
            for row in reader:
                if not self.country2idx_dict.__contains__(row[1]):
                    self.country2idx_dict[row[1]] = idx
                    idx += 1

        self.country_num = idx
        self.idx2country_dict = dict()
        for k, v in self.country2idx_dict.items():
            self.idx2country_dict[v] = k

    def encode(self, country):
        return self.country2idx_dict[country]

    def decode(self, idx):
        return self.idx2country_dict[idx]

    def get_country_size(self):
        return self.country_num



class NameDataset:

    def __init__(self, file_path):
        self.names = []
        self.countries = []
        self.length = 0
        with open(file_path, "rt") as f:
            reader = csv.reader(f)
            for row in reader:
                # 统计名称、语言
                self.names.append(row[0])
                self.countries.append(row[1])
                self.length += 1

    def __getitem__(self, index):
        return self.names[index], self.countries[index]

    def __len__(self):
        return self.length



# 字符编码为数组
def collate_fn(data):
    # 按name长度降序
    data.sort(key=lambda unit: len(unit[0]), reverse=True)
    data_size, max_name_len = len(data), len(data[0][0])  # 降序后，第一个就是长度最大值
    train_file_path = "names_train.csv"
    tokenizer = CountryTokenizer(train_file_path)
    name_seq = torch.zeros(data_size, max_name_len, dtype=torch.long)
    name_len_seq, countries = [], []
    for idx, (name, country) in enumerate(data):
        name_seq[idx, :len(name)] = torch.LongTensor([ord(c) for c in name]) #返回一个相对应的Unicode值， 不足的补0
        name_len_seq.append(len(name))
        countries.append(tokenizer.encode(country))

    return name_seq, torch.LongTensor(name_len_seq), torch.LongTensor(countries)

