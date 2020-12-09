import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--test_data', type=str, default='testdataexample')
    parser.add_argument('-m', '--model', type=str, default='model')

    args = parser.parse_args()
    testData = args.test_data
    model = args.model