function Y_Interp=InterpMfcc(Y,InterpMultiple)
[RowN,ColN]=size(Y);
x=1:ColN;
xp=1:1/InterpMultiple:ColN;
for i=1:RowN
    Y_Interp(i,:)=interp1(Y(i,:),xp,'linear');
end
