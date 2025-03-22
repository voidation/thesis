For SAtest data,
satest-log.txt contains the start time for the session
In the data file for the session, second last column is the time (milliseconds)
To get actual time, need to find the difference between second last column and the millisecond value in satest-log.txt
Then, add this difference to the actual datetime shown to get the datetime value for the satest stuff
Then, compare the satest timestamp to EEG data timestamp and "label" the EEG data with the appropriate scores

1280x800 works for laptop when running satest

