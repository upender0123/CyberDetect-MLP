from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, count, when
from pyspark.ml.feature import MinMaxScaler, VectorAssembler, OneHotEncoder, StringIndexer
import pandas as pd, os

def init_spark(app_name):
    return SparkSession.builder.master("local[*]").appName(app_name).getOrCreate()

def preprocess_data(input_csv, output_path):
    spark = init_spark("CyberDetect-MLP-Preprocessing")
    df = spark.read.csv(input_csv, header=True, inferSchema=True)
    # Handle missing
    for c in df.columns:
        if dict(df.select(count(when(col(c).isNull(), c)).alias('missing')).collect()[0])['missing'] > 0:
            df = df.na.fill({c: df.select(mean(c)).first()[0]}) if df.select(c).dtypes[0][1] in ['double', 'int'] else df.na.fill({c: 'Unknown'})
    # Encode categorical
    cat_cols = [c for c, t in df.dtypes if t == 'string']
    for c in cat_cols:
        indexer = StringIndexer(inputCol=c, outputCol=c+"_idx").fit(df)
        df = indexer.transform(df)
        encoder = OneHotEncoder(inputCols=[c+"_idx"], outputCols=[c+"_vec"])
        df = encoder.fit(df).transform(df)
    # Normalize numeric
    num_cols = [c for c, t in df.dtypes if t in ['double','int']]
    assembler = VectorAssembler(inputCols=num_cols, outputCol="num_features")
    df = assembler.transform(df)
    scaler = MinMaxScaler(inputCol="num_features", outputCol="scaled_features")
    df = scaler.fit(df).transform(df)
    pandas_df = df.toPandas()
    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, "preprocessed.csv")
    pandas_df.to_csv(out_file, index=False)
    print(f"✅ Preprocessed data saved to {out_file}")
    spark.stop()
    return out_file
