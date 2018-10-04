function RF(target)
Len=size(target,1);
len1=round(Len*0.8);
features =target(1:len1-1);
classLabels =target(2:len1);
 
% How many trees do you want in the forest? 
nTrees = 20;
 
% Train the TreeBagger (Decision Forest).
B = TreeBagger(nTrees,features,classLabels, 'Method', 'classification');
 testdata=target(len1+1:Len);
 testpredict= B.predict(testdata);
 predictdata=zeros(size(testpredict,1),1);
 for i=1:size(testpredict,1)
     predictdata(i)=str2double(cell2mat(testpredict(i)));
 end
 mape=100/(Len-len1)*sum(abs(target(len1+1:Len)-predictdata)./target(len1+1:Len));
 rmse=1/(Len-len1)*sqrt((target(len1+1:Len)-predictdata)'*(target(len1+1:Len)-predictdata));
    
 figure
 plot(1:Len,target,'b');
 hold on;
 plot(len1+1:Len,predictdata,'r');
 title(['RF model(mape=', num2str(mape),', rmse=',num2str(rmse),')']);
legend( 'Actual Future Price','Predicted Price');
end