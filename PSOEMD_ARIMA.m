function PSOEMD_ARIMA(target)
imf=emd(target,1);
TT=target-imf(end,:)'+mean(imf(end,:));
Len=size(target,1);
len1=round(Len*0.8);
traindata=TT(1:len1);

nVar=5;            % Number of Decision Variables

VarSize=[1 nVar];   % Size of Decision Variables Matrix

VarMin=-10;         % Lower Bound of Variables
VarMax= 10;         % Upper Bound of Variables


%% PSO Parameters

MaxIt=100;      % Maximum Number of Iterations

nPop=50;        % Population Size (Swarm Size)

% PSO Parameters
w=1;            % Inertia Weight
wdamp=0.99;     % Inertia Weight Damping Ratio
c1=1.5;         % Personal Learning Coefficient
c2=2.0;         % Global Learning Coefficient


% Velocity Limits
VelMax=0.1*(VarMax-VarMin);
VelMin=-VelMax;

%% Initialization

empty_particle.Position=[];
empty_particle.Cost=[];
empty_particle.Velocity=[];
empty_particle.Best.Position=[];
empty_particle.Best.Cost=[];

particle=repmat(empty_particle,nPop,1);

GlobalBest.Cost=inf;

for i=1:nPop
    
    % Initialize Position
    particle(i).Position=unifrnd(VarMin,VarMax,VarSize);
    
    % Initialize Velocity
    particle(i).Velocity=zeros(VarSize);
    
    % Evaluation
    particle(i).Cost=CostFunction(traindata,particle(i).Position);
    
    % Update Personal Best
    particle(i).Best.Position=particle(i).Position;
    particle(i).Best.Cost=particle(i).Cost;
    
    % Update Global Best
    if particle(i).Best.Cost<GlobalBest.Cost
        
        GlobalBest=particle(i).Best;
        
    end
    
end

BestCost=zeros(MaxIt,1);

%% PSO Main Loop

for it=1:MaxIt
    
    for i=1:nPop
        
        % Update Velocity
        particle(i).Velocity = w*particle(i).Velocity ...
            +c1*rand(VarSize).*(particle(i).Best.Position-particle(i).Position) ...
            +c2*rand(VarSize).*(GlobalBest.Position-particle(i).Position);
        
        % Apply Velocity Limits
        particle(i).Velocity = max(particle(i).Velocity,VelMin);
        particle(i).Velocity = min(particle(i).Velocity,VelMax);
        
        % Update Position
        particle(i).Position = particle(i).Position + particle(i).Velocity;
        
        % Velocity Mirror Effect
        IsOutside=(particle(i).Position<VarMin | particle(i).Position>VarMax);
        particle(i).Velocity(IsOutside)=-particle(i).Velocity(IsOutside);
        
        % Apply Position Limits
        particle(i).Position = max(particle(i).Position,VarMin);
        particle(i).Position = min(particle(i).Position,VarMax);
        
        % Evaluation
        particle(i).Cost = CostFunction(traindata,particle(i).Position);
        
        % Update Personal Best
        if particle(i).Cost<particle(i).Best.Cost
            
            particle(i).Best.Position=particle(i).Position;
            particle(i).Best.Cost=particle(i).Cost;
            
            % Update Global Best
            if particle(i).Best.Cost<GlobalBest.Cost
                
                GlobalBest=particle(i).Best;
                
            end
            
        end
        
    end
    
    BestCost(it)=GlobalBest.Cost;
    
%     disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
    
    w=w*wdamp;
    
end

BestSol = GlobalBest;
predictdata=predict_psoarima(target, BestSol.Position,Len-len1,len1);
mape=100/(Len-len1)*sum(abs(target(len1+1:Len)-predictdata)./target(len1+1:Len));
rmse=1/(Len-len1)*sqrt((target(len1+1:Len)-predictdata)'*(target(len1+1:Len)-predictdata));
%% Results
figure;
plot(target,'b');
hold on; plot(len1+1:Len,predictdata,'r');
title(['PSOEMD_ARIMA model(mape=', num2str(mape),', rmse=',num2str(rmse),')']);
legend( 'PSO_ARIMA Result','EMD Result');
end
function z=CostFunction(adata, coe)
coe=coe';
n=size(adata,1);
tdata=zeros(n-3+1,1);
for i=1:size(tdata,1)
    tdata(i)=adata(i:i+1,1)'*coe(1:2,1)+coe(3);
    if(i>2)
        tdata(i)=tdata(i)+(tdata(i-2:i-1,1)-adata(i:i+1,1))'*coe(4:5,1);
    end
end
z=(tdata-adata(3:n,1))'*(tdata-adata(3:n,1))/(n-3+1);
end
function z=predict_psoarima(adata, coe,num,no)
coe=coe';
z=zeros(num,1);
for i=1:num
    z(i)=adata(no-3+1+i:no+i-1,1)'*coe(1:2,1)+coe(3);
    if(i>2)
        z(i)=z(i)+(z(i-2:i-1,1)-adata(no-3+1+i:no+i-1,1))'*coe(4:5,1);
    end
end
end