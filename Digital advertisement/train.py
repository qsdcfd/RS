import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('./dataset.csv')
df.head()

#Average CTR
grouped_label = df.groupby('label').size()
average_ctr = float(grouped_label[1]/grouped_label.sum())
average_ctr

#Process missing Values
df = process_missing_values(df)

#Split into Train and Test data(8:2)

train_test_df = df[['label'] + features]
train, test = train_test_split(train_test_df, test_size = 0.2)

X_train = train[features]
y_train = train['label']

X_test = test[features]
y_test = test['label']

#Build Model

model = lgb.LGBMClassifier(n_estimators=1000,
    learning_rate=0.1,
    num_leaves=100,
    max_depth=15,
    zero_as_missing=True,
    n_jobs=os.cpu_count(),
    objective='binary')

model.fit(X=X_train, y=y_train)

#Evaluate the Trained Model
avg_ctr = average_ctr
prior = log_loss(y_train, [avg_ctr]*len(y_train))

pred = model.predict_proba(X_test)[:, 1]
classifier = log_loss(y_test, pred)

rig = (prior - classifier) / prior

print(f"Baseline: {avg_ctr}")
print(f"RIG: {rig}")

#Dump the model
joblib.dump(model, 'temp/model.pkl')