def read_txt(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    # 去掉每一行末尾的换行符
    lines = [line.strip() for line in lines]
    return lines
