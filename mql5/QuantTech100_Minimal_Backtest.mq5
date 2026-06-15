//+------------------------------------------------------------------+
//| QuantTech100 Minimal Backtest EA                                  |
//| Single-symbol M15 mean reversion. Keep this file simple.          |
//+------------------------------------------------------------------+
#property strict
#property version "1.00"

#include <Trade/Trade.mqh>

input int      EmaPeriod          = 50;
input int      AtrPeriod          = 14;
input double   EntryDistanceAtr   = 2.25;
input double   ExitDistanceAtr    = 0.25;
input double   StopAtr            = 2.0;
input double   TakeProfitR        = 2.0;
input int      MaxBarsInTrade     = 32;
input double   RiskPerTradePct    = 1.25;
input double   DailyStopPct       = 3.0;
input int      MaxTradesPerDay    = 8;
input int      SlippagePoints     = 30;
input ulong    MagicNumber        = 10025001;

CTrade trade;

int emaM15;
int atrM15;
int emaH1;

datetime lastBarTime = 0;
int dayKey = 0;
double dayStartEquity = 0.0;
bool dailyBlocked = false;
int tradesToday = 0;

//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints(SlippagePoints);

   emaM15 = iMA(_Symbol, PERIOD_M15, EmaPeriod, 0, MODE_EMA, PRICE_CLOSE);
   atrM15 = iATR(_Symbol, PERIOD_M15, AtrPeriod);
   emaH1 = iMA(_Symbol, PERIOD_H1, EmaPeriod, 0, MODE_EMA, PRICE_CLOSE);

   if(emaM15 == INVALID_HANDLE || atrM15 == INVALID_HANDLE || emaH1 == INVALID_HANDLE)
   {
      Print("Indicator handle error");
      return INIT_FAILED;
   }

   ResetDay();
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   IndicatorRelease(emaM15);
   IndicatorRelease(atrM15);
   IndicatorRelease(emaH1);
}

//+------------------------------------------------------------------+
void OnTick()
{
   CheckNewDay();
   CheckDailyStop();
   ManageExit();

   if(dailyBlocked)
      return;
   if(tradesToday >= MaxTradesPerDay)
      return;
   if(!NewBarM15())
      return;
   if(HasPosition())
      return;

   CheckBuyEntry();
}

//+------------------------------------------------------------------+
void CheckBuyEntry()
{
   double ema = Indi(emaM15, 1);
   double atr = Indi(atrM15, 1);
   double h1a = Indi(emaH1, 1);
   double h1b = Indi(emaH1, 2);
   double closePrice = iClose(_Symbol, PERIOD_M15, 1);

   if(ema <= 0 || atr <= 0 || h1a <= 0 || h1b <= 0 || closePrice <= 0)
      return;

   double distance = (closePrice - ema) / atr;
   if(distance > -EntryDistanceAtr)
      return;
   if(h1a <= h1b)
      return;

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double stopDistance = atr * StopAtr;
   double sl = NormalizeDouble(ask - stopDistance, _Digits);
   double tp = NormalizeDouble(ask + stopDistance * TakeProfitR, _Digits);
   double lots = LotsFromRisk(stopDistance);

   if(lots <= 0)
      return;

   if(trade.Buy(lots, _Symbol, ask, sl, tp))
   {
      tradesToday++;
      Print("BUY ", _Symbol, " lots=", lots, " sl=", sl, " tp=", tp);
   }
   else
   {
      Print("BUY failed retcode=", trade.ResultRetcode());
   }
}

//+------------------------------------------------------------------+
void ManageExit()
{
   if(!HasPosition())
      return;

   double ema = Indi(emaM15, 1);
   double atr = Indi(atrM15, 1);
   double closePrice = iClose(_Symbol, PERIOD_M15, 1);

   if(ema <= 0 || atr <= 0 || closePrice <= 0)
      return;

   double distance = MathAbs((closePrice - ema) / atr);
   bool meanExit = (distance <= ExitDistanceAtr);

   datetime openTime = (datetime)PositionGetInteger(POSITION_TIME);
   bool timeExit = ((TimeCurrent() - openTime) >= MaxBarsInTrade * PeriodSeconds(PERIOD_M15));

   if(meanExit || timeExit)
      trade.PositionClose(_Symbol);
}

//+------------------------------------------------------------------+
double LotsFromRisk(double stopDistance)
{
   double riskMoney = AccountInfoDouble(ACCOUNT_EQUITY) * RiskPerTradePct / 100.0;
   double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double stepLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   if(riskMoney <= 0 || stopDistance <= 0 || tickSize <= 0 || tickValue <= 0 || stepLot <= 0)
      return 0.0;

   double lossOneLot = stopDistance / tickSize * tickValue;
   if(lossOneLot <= 0)
      return 0.0;

   double lots = riskMoney / lossOneLot;
   lots = MathFloor(lots / stepLot) * stepLot;
   if(lots < minLot)
      lots = minLot;
   if(lots > maxLot)
      lots = maxLot;

   return NormalizeDouble(lots, 2);
}

//+------------------------------------------------------------------+
double Indi(int handle, int shift)
{
   double b[];
   ArraySetAsSeries(b, true);
   if(CopyBuffer(handle, 0, shift, 1, b) != 1)
      return 0.0;
   return b[0];
}

//+------------------------------------------------------------------+
bool NewBarM15()
{
   datetime t = iTime(_Symbol, PERIOD_M15, 0);
   if(t == lastBarTime)
      return false;
   lastBarTime = t;
   return true;
}

//+------------------------------------------------------------------+
bool HasPosition()
{
   if(!PositionSelect(_Symbol))
      return false;
   if((ulong)PositionGetInteger(POSITION_MAGIC) != MagicNumber)
      return false;
   return true;
}

//+------------------------------------------------------------------+
void CheckDailyStop()
{
   if(dayStartEquity <= 0)
      return;

   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double dd = (equity - dayStartEquity) / dayStartEquity;
   if(dd <= -DailyStopPct / 100.0)
   {
      dailyBlocked = true;
      if(HasPosition())
         trade.PositionClose(_Symbol);
   }
}

//+------------------------------------------------------------------+
void CheckNewDay()
{
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   int today = dt.year * 10000 + dt.mon * 100 + dt.day;
   if(today != dayKey)
      ResetDay();
}

//+------------------------------------------------------------------+
void ResetDay()
{
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   dayKey = dt.year * 10000 + dt.mon * 100 + dt.day;
   dayStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   dailyBlocked = false;
   tradesToday = 0;
}
