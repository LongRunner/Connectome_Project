function [ output_args ] = show_act_curves( )
x = 0:1:100;
y = gaussmf(x,[2 50]);
y2 = gaussmf(x,[1.5 47]);

plot(x,y);
hold all;
plot(x,y2);
xlabel('time');
ylabel('true activity');
title('Actual Neural Activity');
end

