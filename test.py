def conv_pattern_extractor(pattern):
    pattern = pattern.replace('x', '*')
    num_blocks = 1
    if '*' in pattern:
        pattern, p2 = pattern.split('*')
        num_blocks = int(p2)
    pattern = pattern.strip()[1:-1]
    p1, p2 = pattern.split(',')
    nb_filters = int(p2)
    if '/' in p1:
        receptive_field, strides = [int(v) for v in p1.split('/')]
    else:
        receptive_field = int(p1)
        strides = 1
    print(nb_filters, receptive_field, strides, num_blocks)


if __name__ == '__main__':
    conv_pattern_extractor('[80/4, 256] x 2')
    conv_pattern_extractor('[80/4, 256]')
    conv_pattern_extractor('[3, 512]')
