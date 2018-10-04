function SVM(target)
Len=size(target,1);
len1=round(Len*0.8);
len1=200;
X=target(1:len1-1);
Y=target(2:len1);
Md1=fitcecoc(X,Y);
Pr=predict(Md1,target(len1+1:len1+100));
figure;
 plot(1:len1,target(1:len1),'b');
 hold on;
plot(len1+1:len1+100,Pr,'r');
end