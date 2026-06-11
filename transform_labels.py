import os


def loadDatadet(infile, k):
    f = open(infile, 'r', encoding='utf-8')  # 增加编码参数避免中文乱码
    sourceInLine = f.readlines()
    dataset = []
    for line in sourceInLine:
        temp1 = line.strip('\n')
        temp2 = temp1.split()  # 去掉括号空格更简洁
        dataset.append(temp2)

    # 关键修复：先转换为浮点数，再替换原列表
    for i in range(len(dataset)):
        # 先提取前k个元素并转换为浮点数
        float_list = []
        for j in range(k):
            try:
                float_list.append(float(dataset[i][j]))
            except ValueError:
                # 处理非数字的异常情况
                print(f"警告：文件 {infile} 第 {i + 1} 行第 {j + 1} 列不是数字，已设为0")
                float_list.append(0.0)
        # 替换原列表的前k个元素（而不是追加后删除）
        dataset[i] = float_list
    f.close()
    return dataset


origin_dataset_path = './data/test_data/val/labels'
output_dir_path = './data/test_data/val/labels'

k = 9
if not os.path.exists(output_dir_path):
    os.makedirs(output_dir_path)  # 改用makedirs，支持创建多级目录

# 获取目录下的文件列表（过滤掉子目录）
dataset = [f for f in os.listdir(origin_dataset_path)
           if os.path.isfile(os.path.join(origin_dataset_path, f))]

for file in dataset:
    inputfile = os.path.join(origin_dataset_path, file)  # 改用os.path.join避免路径拼接错误
    # 检查文件是否为空
    if os.path.getsize(inputfile) == 0:
        outputfile = os.path.join(output_dir_path, file)
        # 清空空文件（保持原有逻辑）
        with open(outputfile, 'w', encoding='utf-8') as output:
            pass
    else:
        one_file_data = loadDatadet(inputfile, k)
        outputfile = os.path.join(output_dir_path, file)

        with open(outputfile, 'w', encoding='utf-8') as output:  # 改用with语句自动关闭文件
            for line in one_file_data:
                # 关键修复：确保line中的元素都是浮点数
                if len(line) < 9:
                    print(f"警告：文件 {inputfile} 某行数据不足{k}个，已跳过")
                    continue

                names = line[0]  # 第一个元素（类别ID）
                tempx = line[1:9:2]  # 提取x坐标：索引1,3,5,7
                tempy = line[2:9:2]  # 提取y坐标：索引2,4,6,8

                # 现在所有数据都是浮点数，可以正常比较
                xmax = max(tempx)
                ymax = max(tempy)
                xmin = min(tempx)
                ymin = min(tempy)

                xcenter = (xmax + xmin) / 2
                ycenter = (ymax + ymin) / 2
                w = xmax - xmin
                h = ymax - ymin

                # 拼接输出内容
                output_line = (f"{names} {round(xcenter, 6)} {round(ycenter, 6)} "
                               f"{round(w, 6)} {round(h, 6)}")
                # 追加原始的9个坐标点
                for i in range(9):
                    output_line += f" {line[i]}"
                output_line += "\n"
                output.write(output_line)