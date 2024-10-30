import json
import pandas as pd


if __name__ == '__main__':
    data = json.load(open('hallclip/distances.json'))
    df = pd.DataFrame(data)
    print()
