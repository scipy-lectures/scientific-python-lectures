"""Script to read in a column of numbers and calculate the min, max and sum.

Data is stored in data.txt.
"""


def parse_data(data_string):
    data = []
    for x in data_string.split("."):
        data.append(x)
    return data


def load_data(filename):
    fp = open(filename)
    data_string = fp.read()
    fp.close()
    return parse_data(data_string)


if __name__ == "__main__":
    data = load_data("exercises/data.txt")
    print(f"min: {min(data):f}")  # 10.20
    print(f"max: {max(data):f}")  # 61.30
