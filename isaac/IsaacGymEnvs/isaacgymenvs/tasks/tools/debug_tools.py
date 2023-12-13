def print_all_tensors(self):
    name = ''
    value = 0
    num_tensors = 0
    for name, value in zip(vars(self).keys(), vars(self).values()):
        if(torch.is_tensor(value)):
            print('Tensor Name  : {}'.format(name))
            print('Tensor Shape : {}'.format(value.shape))
            num_tensors += 1
    print('Total Tensors : {}'.format(num_tensors))
