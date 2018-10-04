function SVR(target)
% Len=size(target,1);
% len1=round(Len*0.8);
len1=200;
X=target(1:len1-1);
Y=target(2:len1);
Md1=fitcecoc(X,Y);
Pr=predict(Md1,target(len1+1:len1+100));
mape=100/100*sum(abs(target(len1+1:len1+100)-Pr)./target(len1+1:len1+100));
rmse=1/100*sqrt((target(len1+1:len1+100)-Pr)'*(target(len1+1:len1+100)-Pr));
figure;
 plot(1:len1+100,target(1:len1+100),'b');
 hold on;
plot(len1+1:len1+100,Pr,'r');
title(['SVR model(mape=', num2str(mape),', rmse=',num2str(rmse),')']);
legend( 'Actual Future Price','Predicted Price');
end