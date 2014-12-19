def print_b():
    "Prints b."
    print 'b'

def print_a():
    "Prints a."
    print 'a'

# print_b() runs on import
print_b()

if __name__ == '__main__':
    # print_a() is only executed when the module is run directly.
    print_a()
