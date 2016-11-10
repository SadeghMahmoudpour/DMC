import pandas as pd


def convert_to_csv(path, inputFileName, outputFileName):
    inputPath = '{path}/{inputFileName}'.format(path=path, inputFileName=inputFileName)
    outputPath = '{path}/{outputFileName}'.format(path=path, outputFileName=outputFileName)

    outputFile = open(outputPath, 'w', encoding='utf-8')

    with open(inputPath, 'r', encoding='utf-8') as inputFile:
        for line in inputFile:
            csvline = line.replace(';', ',')
            outputFile.write(csvline)

        outputFile.close()


def get_orders_train():
    return pd.read_csv('datasets/orders_train.txt', sep=';', dtype={'productGroup':object, 'colorCode':object, 'colorCode':object})


def get_train_data():
    return get_orders_train()[680001:860001]


def get_test_data():
    return get_orders_train()[2000001:2200001]


def clean_by_quantity(df):
    df = df[df['quantity'] > 0]
    df = df[df['quantity'] < 6]
    df.reset_index(drop=True, inplace=True)
    return df


def clean_by_rrp(df):
    df['rrp'] = df['rrp'].fillna(df['price'] / df['quantity'])
    return df

def clean_by_productGroup(df):

    return df


def clean_df(df):
    return clean_by_rrp(clean_by_quantity(df))


def get_df_attributes(df, attr):
    return df.loc[:, attr]


def get_train_attrs(df):
    attrs = ['articleID', 'colorCode', 'sizeCode', 'quantity', 'price', 'rrp', 'voucherAmount', 'paymentMethod']
    return get_df_attributes(df, attrs)


def value_transform(df):
    from sklearn.preprocessing import LabelEncoder
    var_mod = list(df.columns.values)
    le = LabelEncoder()
    for i in var_mod:
        df[i] = le.fit_transform(df[i])
    return df


def process_df(df):
    testdata = clean_df(df)
    testcols = get_train_attrs(testdata)
    testcols = value_transform(testcols)
    return testcols, testdata['returnQuantity']


def compute_error(result, real):
    return sum(abs(result-real))


def svm_learn(X, y):
    from sklearn import svm

    clf = svm.SVC()

    clf.fit(X, y)
    return clf


def gradient_boosting_classifier(X, y):
    from sklearn.ensemble import GradientBoostingClassifier

    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    max_depth = 1, random_state = 0).fit(X, y)

    return clf


def xgboost_classifier(X, y):
    from xgboost import XGBClassifier

    model = XGBClassifier()
    model.fit(X, y)
    return model

