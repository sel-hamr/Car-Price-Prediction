import sys
from model.model import train,predict

def check_args(argv: list["str"]) -> None :
    if len(argv) > 1:
        if argv[1] == '-L':
            if len(argv) > 2:
                train(argv[2])
            else:
                raise ValueError("'-L' option provided without path.")
        else:
            print(f"Script executed with argument: {argv[1]}")
    elif len(argv) == 1:
        while True:
            predict()
            if input("Do you want to predict another car? (y/n): ") == "n":
                break
    else:
        raise ValueError("Invalid number of arguments provided.")

def main(argv: list["str"]) -> None:
    try:
        check_args(argv)

    except ValueError as error:
        print(f"Error: {error}")

if __name__ == "__main__":
    main(sys.argv)