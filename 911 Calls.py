import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as s

# reading the dataset
df = pd.read_csv('911.csv')
# check the info of this dataset
# print(df.info())

# check the head
# print(df.head())
#        zip                    title            timeStamp                twp  \
# 0  19525.0   EMS: BACK PAINS/INJURY  2015-12-10 17:10:52        NEW HANOVER
# 1  19446.0  EMS: DIABETIC EMERGENCY  2015-12-10 17:29:21  HATFIELD TOWNSHIP
# 2  19401.0      Fire: GAS-ODOR/LEAK  2015-12-10 14:39:21         NORRISTOWN
# 3  19401.0   EMS: CARDIAC EMERGENCY  2015-12-10 16:47:36         NORRISTOWN
# 4      NaN           EMS: DIZZINESS  2015-12-10 16:56:52   LOWER POTTSGROVE
#
#                          addr  e
# 0      REINDEER CT & DEAD END  1
# 1  BRIAR PATH & WHITEMARSH LN  1
# 2                    HAWS AVE  1
# 3          AIRY ST & SWEDE ST  1
# 4    CHERRYWOOD CT & DEAD END  1
#


# q1: what are the top five zipcodes for the 911 calls? Lets see


# print(df['zip'].value_counts()) #this line willl give us the occurence or count of all the zip codes. how many times each zip
# occures
# to get the top five only we just add head(5)
print(df['zip'].value_counts().head(5))

print('\n\n\n')

# q2: what are the top five township for the 911 calls
# again we use the value_counts() to get the occurence of all then we add head(5) to get the top 5 as
print(df['twp'].value_counts().head(5))

print('\n\n\n')

# q3: find how many unique title codes are there in the column 'title':
# print(df['title'].unique())
# this gives us all the unique entries now we can count the length of this
# print(len(df['title'].unique()))
# or we use nunique()
print(df['title'].nunique())

print('\n\n\n')

# q4: creating new features
# in the title column there are features and 'reasons/departments' specified before the title code.
# these are ems, fire and traffic. use .apply() with custom lambda expression to create a new col  called 'reason'
# that contains this string value
# print(df['title'].head(5))
# 0     EMS: BACK PAINS/INJURY
# 1    EMS: DIABETIC EMERGENCY
# 2        Fire: GAS-ODOR/LEAK
# 3     EMS: CARDIAC EMERGENCY
# 4             EMS: DIZZINESS

# here the col contains the reason as well but we dont want this reason in the new col
# we only want the main reason
# eg if col contains  EMS: BACK PAINS/INJURY the reason col should only have EMS
# to do this we first grab the title col and then the first value from this col
# print(df['title'].iloc[0])
# EMS: BACK PAINS/INJURY
# this is returned and now we store this in a string to work on it
# a = df['title'].iloc[0]
# now we only need the EMS and no anything after this so we split the string
# print(a.split(':'))
# by default split work simply, it splits the string by the space as
# this is a string becomes 'this','is','a','string'  now you can grab these words from it usig the index
# but if you want to split the string accorind to some other character then yuo have to pass that as
# string.split(':') now the string will be split where : occures this is waht we did
# b = a.split(':')
# print(b[0]) EMS this is the output
# now we use this for the lambda expression

df['reason'] = df['title'].apply(lambda title: title.split(':')[0])
# print(df['reason'])
# 0             EMS
# 1             EMS
# 2            Fire
# 3             EMS
# 4             EMS
# 5             EMS
# 6             EMS
# 7             EMS
# 8             EMS
# 9         Traffic
# 10        Traffic
# 11        Traffic
# 12        Traffic
# 13        Traffic
# 14        Traffic
# 15        Traffic
# 16            EMS
# 17            EMS
# 18            EMS
# 19        Traffic
# 20        Traffic
# 21        Traffic
# 22           Fire
# 23        Traffic
# 24        Traffic
# 25            EMS
# 26            EMS
# 27           Fire
# 28        Traffic
# 29        Traffic
#            ...
# 326395       Fire
# 326396        EMS
# 326397        EMS
# 326398    Traffic
# 326399    Traffic
# 326400    Traffic
# 326401    Traffic
# 326402        EMS
# 326403       Fire
# 326404    Traffic
# 326405       Fire
# 326406       Fire
# 326407    Traffic
# 326408    Traffic
# 326409    Traffic
# 326410    Traffic
# 326411    Traffic
# 326412        EMS
# 326413       Fire
# 326414    Traffic
# 326415    Traffic
# 326416    Traffic
# 326417        EMS
# 326418       Fire
# 326419        EMS
# 326420    Traffic
# 326421    Traffic
# 326422    Traffic
# 326423    Traffic
# 326424       Fire
# Name: reason, Length: 326425, dtype: object


# q5: what are the most common reason to call 911? Tell this based on this new column 'reason'.
print(df['reason'].value_counts().head(1))

# Now headed to Visualisation
# q6 create a seasborn countplot to visualise the reason column

# s.countplot(x= 'reason', data = df)
# here x means the x axis which we said reason and there is no y axis in the countplot it simply counts the occurence
# OF ALL the entries in that column
# and then data = dataframe which is df
# plt.show()


print('\n\n\n')

# q7: what is the datatype of the objects in the timestamp column
# to do so we will grab the first row of the timeStamp col and then check its datatype
print(type(df['timeStamp'].iloc[0]))

print('\n\n\n')

# q8: now convert these timeStamp which are string into datetime objects
# this can be done by using to_datatime()
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
print(type(df['timeStamp'].iloc[0]))

# q9: now use this newly created datetime column to create 3 new cols
# hour, month and day of week and day of the waek
# t = df['timeStamp'].iloc[0]
# grabbing the first row
# print(t.month) the month of the first row
# print(t.hour) the hour
# print(t.day) the day of the month

# creating col month
df['month'] = df['timeStamp'].apply(lambda timeStamp: timeStamp.month)
# creating hour
df['hour_of_day'] = df['timeStamp'].apply(lambda timeStamp: timeStamp.hour)
# creating day of the month ie date
df['date'] = df['timeStamp'].apply(lambda timeStamp: timeStamp.day)
# creating day of the week which gives us a nuumber 0-6 which we will convert into a day string using a dictionary
# creating a dict for day of the week
day = {0: 'mon', 1: 'tue', 2: 'wed', 3: 'thur', 4: 'fri', 5: 'sat', 6: 'sun'}
df['day_of_week'] = df['timeStamp'].apply(lambda timeStamp: day[timeStamp.dayofweek])

print(df.head(10))

# q10: now use seaborn to create countplot of the day of the week with hue based on the reason column
# s.countplot(x = 'day_of_week',data = df)
# plt.show()
# without hue it will show the count for each day of the week ie count for mon,tue,wed etc
# but with hue it will also show reason count for each day as well
# which means it will show count for mon with reason EMS, Fire, Traffic etc
# then it will show count for tue with reason EMS, Fire, Traffic etc and so on
# s.countplot(x = 'day_of_week',data= df,hue='reason')
# plt.show()



# now we'll do the same thing but with months
# s.countplot(x = 'month',data=df,hue='reason')
# plt.show()

print('\n\n\n')

# create a groupby object ussing the col month and use the agg function count
bymonth = df.groupby('month').count()
# bymonth is a dataframe
# print(bymonth.head())
# bymonth.plot('lat')
# plt.show()
# we can use dataframe to build plots ehich we already know


# q11: we have created a date column above. now use this date column to create a countplot to show calls each day of the month
# s.countplot(x = 'date',data=df)
# plt.show()

# now we will create a groupby object to group by according to date colm

dategroup = df.groupby('date').count()
# print(dategroup['lat'])
# this shows the number of calls received per day of the month
# date
# 1     10111
# 2     12364
# 3     11091
# 4     10064
# 5     10450
# 6     10555
# 7     10687
# 8     10044
# 9     10671
# 10    10411
# 11    10837
# 12    10773
# 13    11206
# 14    10875
# 15    11302
# 16    10942
# 17    11106
# 18    10564
# 19    11102
# 20    10513
# 21    11017
# 22    10969
# 23    11544
# 24    11260
# 25    10503
# 26     9958
# 27     9946
# 28    10440
# 29     9753
# 30     9308
# 31     6059
# Name: lat, dtype: int64



# dategroup['lat'].plot()
# plt.show()


# Creating heatmaps now
dayh = df.groupby(by=['day_of_week', 'hour_of_day']).count()['reason'].unstack()
# this will show count of calls at each hour of the day
# eg calls at first hour on mon, then second hour on mon and so on
# then 1st hour on tue then second on tue and so on
# the unstack method will create aa matrix of all this
# this creates a  dataframe with multi level index and which shows the above infomation



# print(dayh)
#                           lat   lng  desc   zip  title  timeStamp   twp  addr  \
#day_of_week  hour_of_day
# fri         0             896   896   896   798    896        896   896   896
#             1             789   789   789   694    789        789   786   789
#             2             701   701   701   640    701        701   701   701
#             3             644   644   644   583    644        644   644   644
#             4             633   633   633   587    633        633   632   633
#             5             786   786   786   691    786        786   785   786
#             6            1286  1286  1286  1081   1286       1286  1284  1286
#             7            2087  2087  2087  1833   2087       2087  2086  2087
#             8            2487  2487  2487  2187   2487       2487  2487  2487
#             9            2570  2570  2570  2266   2570       2570  2570  2570
#             10           2727  2727  2727  2409   2727       2727  2726  2727
#             11           2889  2889  2889  2563   2889       2889  2888  2889
#             12           3042  3042  3042  2710   3042       3042  3042  3042
#             13           3210  3210  3210  2779   3210       3210  3210  3210
#             14           3290  3290  3290  2894   3290       3290  3288  3290
#             15           3562  3562  3562  3062   3562       3562  3561  3562
#             16           3726  3726  3726  3226   3726       3726  3726  3726
#             17           3596  3596  3596  3090   3596       3596  3595  3596
#             18           2858  2858  2858  2507   2858       2858  2857  2858
#             19           2562  2562  2562  2303   2562       2562  2562  2562
#             20           2205  2205  2205  1936   2205       2205  2204  2205
#             21           1916  1916  1916  1691   1916       1916  1916  1916
#             22           1765  1765  1765  1558   1765       1765  1764  1765
#             23           1396  1396  1396  1236   1396       1396  1392  1396
# mon         0             931   931   931   832    931        931   931   931
#             1             732   732   732   643    732        732   730   732
#             2             663   663   663   590    663        663   663   663
#             3             585   585   585   512    585        585   585   585
#             4             683   683   683   612    683        683   683   683
#             5             862   862   862   768    862        862   862   862
# ...                       ...   ...   ...   ...    ...        ...   ...   ...
# tue         18           2918  2918  2918  2517   2918       2918  2916  2918
#             19           2283  2283  2283  2053   2283       2283  2282  2283
#             20           1960  1960  1960  1747   1960       1960  1958  1960
#             21           1660  1660  1660  1450   1660       1660  1659  1660
#             22           1329  1329  1329  1185   1329       1329  1329  1329
#             23           1025  1025  1025   894   1025       1025  1025  1025
# wed         0             805   805   805   743    805        805   804   805
#             1             738   738   738   664    738        738   738   738
#             2             620   620   620   558    620        620   619   620
#             3             626   626   626   574    626        626   624   626
#             4             560   560   560   518    560        560   560   560
#             5             798   798   798   700    798        798   798   798
#             6            1382  1382  1382  1144   1382       1382  1381  1382
#             7            2319  2319  2319  1989   2319       2319  2319  2319
#             8            2751  2751  2751  2387   2751       2751  2751  2751
#             9            2744  2744  2744  2441   2744       2744  2744  2744
#             10           2691  2691  2691  2402   2691       2691  2690  2691
#             11           2749  2749  2749  2458   2749       2749  2749  2749
#             12           2946  2946  2946  2629   2946       2946  2945  2946
#             13           2838  2838  2838  2539   2838       2838  2838  2838
#             14           3095  3095  3095  2730   3095       3095  3094  3095
#             15           3211  3211  3211  2833   3211       3211  3211  3211
#             16           3413  3413  3413  2968   3413       3413  3412  3413
#             17           3435  3435  3435  3005   3435       3435  3434  3435
#             18           2782  2782  2782  2428   2782       2782  2782  2782
#             19           2296  2296  2296  2022   2296       2296  2296  2296
#             20           2064  2064  2064  1837   2064       2064  2062  2064
#             21           1682  1682  1682  1501   1682       1682  1678  1682
#             22           1422  1422  1422  1242   1422       1422  1421  1422
#             23           1103  1103  1103   982   1103       1103  1100  1103
#
#                             e  reason  month  date
# day_of_week hour_of_day
# fri         0             896     896    896   896
#             1             789     789    789   789
#             2             701     701    701   701
#             3             644     644    644   644
#             4             633     633    633   633
#             5             786     786    786   786
#             6            1286    1286   1286  1286
#             7            2087    2087   2087  2087
#             8            2487    2487   2487  2487
#             9            2570    2570   2570  2570
#             10           2727    2727   2727  2727
#             11           2889    2889   2889  2889
#             12           3042    3042   3042  3042
#             13           3210    3210   3210  3210
#             14           3290    3290   3290  3290
#             15           3562    3562   3562  3562
#             16           3726    3726   3726  3726
#             17           3596    3596   3596  3596
#             18           2858    2858   2858  2858
#             19           2562    2562   2562  2562
#             20           2205    2205   2205  2205
#             21           1916    1916   1916  1916
#             22           1765    1765   1765  1765
#             23           1396    1396   1396  1396
# mon         0             931     931    931   931
#             1             732     732    732   732
#             2             663     663    663   663
#             3             585     585    585   585
#             4             683     683    683   683
#             5             862     862    862   862
# ...                       ...     ...    ...   ...
# tue         18           2918    2918   2918  2918
#             19           2283    2283   2283  2283
#             20           1960    1960   1960  1960
#             21           1660    1660   1660  1660
#             22           1329    1329   1329  1329
#             23           1025    1025   1025  1025
# wed         0             805     805    805   805
#             1             738     738    738   738
#             2             620     620    620   620
#             3             626     626    626   626
#             4             560     560    560   560
#             5             798     798    798   798
#             6            1382    1382   1382  1382
#             7            2319    2319   2319  2319
#             8            2751    2751   2751  2751
#             9            2744    2744   2744  2744
#             10           2691    2691   2691  2691
#             11           2749    2749   2749  2749
#             12           2946    2946   2946  2946
#             13           2838    2838   2838  2838
#             14           3095    3095   3095  3095
#             15           3211    3211   3211  3211
#             16           3413    3413   3413  3413
#             17           3435    3435   3435  3435
#             18           2782    2782   2782  2782
#             19           2296    2296   2296  2296
#             20           2064    2064   2064  2064
#             21           1682    1682   1682  1682
#             22           1422    1422   1422  1422
#             23           1103    1103   1103  1103
#
# [168 rows x 12 columns]
# s.heatmap(dayh)
# plt.show()

# this heatmap will visualize the calls at each hour a day

