# -*- coding: utf-8 -*-"""
"""
Built the profile and their testing
This script is not based on any dataset, it's a dummy
"""
import os
import argparse
import datetime
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
import boto3
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType
from slackclient import SlackClient

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--generatedon", '-g', metavar="generatedon",
                    type=str, default=datetime.date.today().strftime('%Y%m%d'),
                    help="versionning of datasets")
PARSER.add_argument("--history", '-hi', metavar="history",
                    type=int, default=24,
                    help="history to be consider to built the profiles")
PARSER.add_argument("--profile", '-p', metavar="profile",
                    default=['city'],
                    help="list of column to compare aggregated data to. ex:city",
                    type=lambda s: [str(item) for item in s.split(',')])
PARSER.add_argument("--peer", '-pe', metavar="peer",
                    default=['population_size'],
                    help="list of column to build peer groups profile with ex:population_size, country",
                    type=lambda s: [str(item) for item in s.split(',')])
CONF = PARSER.parse_args()
GENERATED_ON = CONF.generatedon
HISTORY = CONF.history
PROFILE = CONF.profile
PEER = CONF.peer
PROJECTNAME = 'Anomaly detection IQR'
SCRIPTNAME = 'weekly_anomalie_detection.py'
SRC_DOMAIN_PATH = 's3://somepathtotheinputdata/'
TGT_DOMAIN_PATH = 's3://somepathtotheoutputfoler/'

def get_bearer_token(parameterstore_key=None, encrypted=True):
    """get slack token"""
    if not parameterstore_key:
        parameterstore_key = os.environ.get('SSM_STORE_SLACK_TOKEN_KEY')
    ssm = boto3.client('ssm', region_name='eu-west-1')
    parameter = ssm.get_parameter(Name=parameterstore_key,
                                  WithDecryption=encrypted)
    token = parameter['Parameter']['Value']
    return token

SLACK_TOKEN = get_bearer_token(parameterstore_key='/Slack/key_to_private_slack')

def send_to_slack(message, target_channel="channelcode"):
    """ send to slack function """
    try:
        message = str(message)
        print(message)
        slackclient = SlackClient(SLACK_TOKEN)
        slackclient.api_call(
            "chat.postMessage",
            channel=target_channel,
            text="%s: %s- %s"%(PROJECTNAME, SCRIPTNAME, message))
    except Exception as general_exception:
        print("cannot send on slack")
        print(general_exception)

def run_etl(raw_data, generated_on, profile_start_date, profile_end_date, test_start_date, test_end_date, path):
    """ Runs all steps needed to build and test weekly profiles """
    sel_cols = PROFILE + ['ppm']# ppm is the value of the concentration
    train = raw_data\
        .filter(F.col('timestamp_measurement') >= profile_start_date)\
        .filter(F.col('timestamp_measurement') <= profile_end_date)\
        .withColumn('timestamp_monday', F.date_trunc('week', F.to_timestamp("timestamp_measurement", "yyyy-MM-dd")))# rounding up weeks to mondays
    # TRAIN: group the data per relevant profiles and filter when profile doesn't contains engouh data points
    sel_cols = PROFILE + ['ppm', 'timestamp_monday']
    instance_profile = train.select(sel_cols)\
        .groupBy(PROFILE + ['timestamp_monday'])\
        .count()\
        .groupBy(PROFILE)\
        .agg(F.count('count').alias('instance_count'))
    train_weekly = train.join(instance_profile.filter(F.col('instance_count') < 5), PROFILE, how='left_anti')\
        .groupBy(PROFILE + ['timestamp_monday']).agg(F.count('ppm').alias('freq'), F.sum('ppm').alias('ppm_sum'))\
        .cache()
    # TEST
    test_weekly = raw_data\
    .filter(F.col('timestamp_measurement') >= test_start_date)\
    .filter(F.col('timestamp_measurement') <= test_end_date)\
    .withColumn('timestamp_monday', F.date_trunc('week', F.to_timestamp("timestamp_measurement", "yyyy-MM-dd")))\
    .groupBy(PROFILE + ['timestamp_monday'])\
    .agg(F.count('ppm').alias('freq'), F.sum('ppm').alias('ppm_sum'))\
    .cache()
    if len(PEER[0]) > 0:# if peer profile are built, consider PEER group as groupby group
        profile_gb = PEER
    else:
        profile_gb = PROFILE
    # BUILDING PROFILE
    for profile_type in ['freq', 'ppm_sum']:# compute profile for outliers in term of total amount per week and number of measurments
        # profile are IQR range profiles
        w_profile = Window.partitionBy(profile_gb)
        quantiles = F.expr('percentile_approx(%s, array(0.25, 0.75))'%profile_type)
        sel_cols = profile_gb + [profile_type]
        df_profile = train_weekly.select(sel_cols)\
            .withColumn('instance_count', F.count(profile_type).over(w_profile))\
            .filter(F.col("instance_count") >= 5)\
            .withColumn('quantiles', quantiles.over(w_profile))\
            .select(profile_gb + ['quantiles']).distinct()\
            .withColumn('quantile1', F.round(F.col('quantiles')[0], 2))\
            .withColumn('quantile3', F.round(F.col('quantiles')[1], 2))\
            .withColumn('iqr', F.round(F.col('quantile3')-F.col('quantile1'), 2))\
            .withColumn('minimum_15', F.round(F.col('quantile1')-1.5*F.col('iqr'), 2))\
            .withColumn('maximum_15', F.round(F.col('quantile3')+1.5*F.col('iqr'), 2))\
            .withColumn('minimum_3', F.round(F.col('quantile1')-3*F.col('iqr'), 2))\
            .withColumn('maximum_3', F.round(F.col('quantile3')+3*F.col('iqr'), 2))\
            .select(profile_gb + ['quantile1', 'iqr', 'quantile3', 'minimum_15', 'maximum_15', 'minimum_3', 'maximum_3'])
        # Write the profile to unload memory
        start = dt.strftime(dt.strptime(profile_start_date, "%Y-%m-%d"), "%Y%m%d")
        end = dt.strftime(dt.strptime(profile_end_date, "%Y-%m-%d"), "%Y%m%d")
        df_profile.write.mode('overwrite').parquet(TGT_DOMAIN_PATH + 'generated_on={0}/profiles/weekly_{1}_{2}/{3}'\
                                                    .format(generated_on, path, profile_type, '{0}_{1}_{2}_to_{3}'\
                                                        .format('_'.join(profile_gb), HISTORY, start, end)))
        # TEST PROFILE
        test_with_label = test_weekly.join(df_profile, profile_gb, how='left')
        test_with_label_no_profiles = test_with_label.filter(F.col('iqr').isNull())
        test_with_label_profiles = test_with_label.filter(F.col('iqr').isNotNull())
        # handling data set with no profiles
        test_with_label_no_profiles = test_with_label_no_profiles.withColumn('prediction15', F.lit('unknown')).withColumn('prediction3', F.lit('unknown'))
        # handling data set with profiles
        exp_prediction15 = F.when(((F.col(profile_type) >= F.col('minimum_15')) & (F.col(profile_type) <= F.col('maximum_15'))), F.lit('normal'))\
            .otherwise(F.lit('anomaly'))
        exp_prediction3 = F.when(((F.col(profile_type) >= F.col('minimum_3')) & (F.col(profile_type) <= F.col('maximum_3'))), F.lit('normal')
                                ).otherwise(F.lit('anomaly'))
        test_with_label_profiles = test_with_label_profiles.withColumn('prediction15', exp_prediction15)\
            .withColumn('prediction3', exp_prediction3)
        # union and write
        start = dt.strftime(dt.strptime(test_start_date, "%Y-%m-%d"), "%Y%m%d")
        end = dt.strftime(dt.strptime(test_end_date, "%Y-%m-%d"), "%Y%m%d")
        test_data = test_with_label_no_profiles.union(test_with_label_profiles)\
            .withColumn('profile', F.lit('weekly_{0}_{1}_{2}_{3}'.format(path, '_'.join(profile_gb), profile_type, HISTORY)))\
            .withColumn('key', F.concat_ws('_', *PROFILE))\
            .select('city', 'timestamp_monday', 'key', 'freq', 'ppm_sum',
                    'quantile1', 'iqr', 'quantile3', 'minimum_15', 'maximum_15', 'minimum_3', 'maximum_3', 'prediction15', 'prediction3', 'profile')
        for col in ['freq', 'ppm_sum', 'quantile1', 'iqr', 'quantile3', 'minimum_15', 'maximum_15', 'minimum_3', 'maximum_3']:
            test_data = test_data.withColumn(col, F.col(col).cast(IntegerType()))
        test_data.write.mode('overwrite').parquet(TGT_DOMAIN_PATH + 'generated_on={0}/test/weekly_{1}_{2}/{3}'\
                                                    .format(generated_on, path, profile_type, '{0}_{1}_{2}_to_{3}'
                                                            .format('_'.join(profile_gb), HISTORY, start, end)))
        # test_data can be analyse in term of how many outliers where flagged, and the profiling can be adjusted accordingly
        return

def main():
    """ entry point, triggers ETL tasks.
    """
    spark = SparkSession.builder.appName('PSD2').getOrCreate()
    send_to_slack('Starting weekly profiles ' + ', '.join(str(x) for x in PROFILE + PEER))
    test_end_date_global = dt.strptime('2021-04-26', "%Y-%m-%d")
    raw_data = spark.read.parquet(*[SRC_DOMAIN_PATH + '/generated_on={0}/tropomi_sensors_values_eu/*.parquet'.format(GENERATED_ON)]).cache()
    try:
        for sensor_measured in ['NO2', 'CO2']:
            profile_end_date = '2021-02-01'
            profile_start_date = dt.strftime(dt.strptime(profile_end_date, "%Y-%m-%d") - relativedelta(months=HISTORY), "%Y-%m-%d")
            data_type = raw_data.filter(F.col(sensor_measured) == 1).cache()
            while dt.strptime(profile_end_date, "%Y-%m-%d") < test_end_date_global:
                test_start_date = dt.strftime(dt.strptime(profile_end_date, "%Y-%m-%d") + relativedelta(days=1), "%Y-%m-%d")
                test_end_date = dt.strftime(dt.strptime(profile_end_date, "%Y-%m-%d") + relativedelta(days=7), "%Y-%m-%d")
                run_etl(data_type, GENERATED_ON, profile_start_date, profile_end_date, test_start_date, test_end_date, sensor_measured)
                profile_end_date = test_end_date
                profile_start_date = dt.strftime(dt.strptime(profile_end_date, "%Y-%m-%d") - relativedelta(months=HISTORY), "%Y-%m-%d")

    except Exception as general_exception:
        send_to_slack('Error during step')
        send_to_slack(general_exception)
        raise
    spark.stop()
if __name__ == '__main__':
    main()
