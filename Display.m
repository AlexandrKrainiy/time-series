function Display(name, target)
Len=size(target,1);
len1=floor(Len*0.95);
if (strcmp(name,'ARIMA'))
    model =arima('Constant',0,'D',1,'Seasonality',12,'MALags',1,'SMALags',12);
    [fit, VarCov] = estimate(model,target(len1+1:end),'Y0',target(1:len1));
    %res=infer(fit,target(len1+1:end),'Y0',target(1:len1));
    %arimaPredictions=res;
    [arimaPredictions YMSE] = forecast(fit,Len-len1,'Y0',target(1:len1));
    figure
    mape=100/(Len-len1)*sum(abs(target(len1+1:Len)-arimaPredictions)./target(len1+1:Len));
    rmse=1/(Len-len1)*sqrt((target(len1+1:Len)-arimaPredictions)'*(target(len1+1:Len)-arimaPredictions));
    plot(1:Len,target,'b');
    hold on;
    plot(len1+1:Len,arimaPredictions,'r');
    legend( 'Actual Future Price','Predicted Price');
    title(['ARIMA model(mape=', num2str(mape),', rmse=',num2str(rmse),')']);
end
if(strcmp(name,'ANN'))
    ANN(target,target);
end
 if(strcmp(name,'SVR'))
    SVR(target);
 end
  if(strcmp(name,'RF'))
    RF(target);
  end
  if(strcmp(name,'PSO'))
    PSO(target);
  end
 if(strcmp(name,'PSOARIMA'))
    PSOARIMA(target);
 end
 if(strcmp(name,'PSOSVR'))
    PSOSVR(target);
 end
  if(strcmp(name,'PSORF'))
    PSORF(target);
  end
 if(strcmp(name,'EMD'))
    emd(target,0);
 end
 if(strcmp(name,'PSOEMD_ARIMA'))
    PSOEMD_ARIMA(target);
 end
 if(strcmp(name,'PSOEMD_SVR'))
    PSOEMD_SVR(target);
 end
 if(strcmp(name,'PSOEMD_RF'))
    PSOEMD_RF(target);
 end
 if(strcmp(name,'EMD_ANN'))
    EMD_ANN(target,target);
 end
end