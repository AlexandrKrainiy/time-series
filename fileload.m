function [open, High, Low, close,volume,Adj]=fileload(filename)
data=csvread(filename,1,1);
open=data(:,1);
High=data(:,2);
Low=data(:,3);
close=data(:,4);
volume=data(:,5);
Adj=data(:,6);
end