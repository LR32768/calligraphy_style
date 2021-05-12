

def add_decimal(num0, num1):
    bit_end = int(num0[-1]) + num1
    return ('3' + str(bit_end))


def add_hexadecimal(num0, num1):
    result = hex(int(num0, 16) + num1)
    return result


def filenum_to_fontcode(filenum):
    filenum_base = 873
    fontcode_base = ['0x81', '30', '0x81', '30']
    fontcode = [None, None, None, None]

    delta = filenum - filenum_base

    carry_0 = delta % 10
    remain_0 = delta // 10
    
    carry_1 = remain_0 % 126
    remain_1 = remain_0 // 126

    carry_2 = remain_1 % 10
    remain_2 = remain_1 // 10

    carry_3 = remain_2 % 126
    assert carry_3 < 126

    fontcode[3] = add_decimal(fontcode_base[3], carry_0)  # 最低第1位
    fontcode[2] = add_hexadecimal(fontcode_base[2], carry_1)  # 最低第2位
    fontcode[1] = add_decimal(fontcode_base[1], carry_2)  # 最低第3位
    fontcode[0] = add_hexadecimal(fontcode_base[0], carry_3)  # 最低第4位

    fontcode_str = fontcode[0][-2:] + fontcode[1] + fontcode[2][-2:] + fontcode[3]

    return fontcode, fontcode_str

filenum = 13383

fontcode, fontcode_str = filenum_to_fontcode(filenum)
print(fontcode)
print(fontcode_str)