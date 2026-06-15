//+------------------------------------------------------------------+
//| QuantTech100 Compact Backtest EA                                  |
//| Single symbol. Attach/test on US100 or XAUUSD, timeframe M15.     |
//+------------------------------------------------------------------+
#property strict
#property version "1.00"

#include <Trade/Trade.mqh>

input double RiskPerTradePct = 1.25;
input double DailyStopPct = 3.0;
input int MaxTradesPerDay = 8;
input ulong MagicNumber = 10025001;

CTrade trade;
int emaM15 = INVALID_HANDLE;
int atrM15 = INVALID_HANDLE;
int emaH1 = INVALID_HANDLE;
datetime lastBar = 0;
int dayKey = 0;
double dayStartEquity = 0.0;
bool dailyBlocked = false;
int tradesToday = 0;

int OnInit()
{
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints(30);
   emaM15 = iMA(_Symbol, PERIOD_M15, 50, 0, MODE_EMA, PRICE_CLOSE);
   atrM15 = iATR(_Symbol, PERIOD_M15, 14);
   emaH1 = iMA(_Symbol, PERIOD_H1, 50, 0, MODE_EMA, PRICE_CLOSE);
   if(emaM15 == INVALID_HANDLE || atrM15 == INVALID_HANDLE || emaH1 == INVALID_HANDLE)
      return INIT_FAILED;
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   dayKey = dt.year * 10000 + dt.mon * 100 + dt.day;
   dayStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   if(emaM15 != INVALID_HANDLE) IndicatorRelease(emaM15);
   if(atrM15 != INVALID_HANDLE) IndicatorRelease(atrM15);
   if(emaH1 != INVALID_HANDLE) IndicatorRelease(emaH1);
}

void OnTick()
{
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   int today = dt.year * 10000 + dt.mon * 100 + dt.day;
   if(today != dayKey)
   {
      dayKey = today;
      dayStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);
      dailyBlocked = false;
      tradesToday = 0;
   }

   if(dayStartEquity > 0.0)
   {
      double dd = (AccountInfoDouble(ACCOUNT_EQUITY) - dayStartEquity) / dayStartEquity;
      if(dd <= -DailyStopPct / 100.0)
      {
         dailyBlocked = true;
         if(PositionSelect(_Symbol) && (ulong)PositionGetInteger(POSITION_MAGIC) == MagicNumber)
            trade.PositionClose(_Symbol);
      }
   }

   double emaBuf[3], atrBuf[3], h1Buf[3];
   ArraySetAsSeries(emaBuf, true);
   ArraySetAsSeries(atrBuf, true);
   ArraySetAsSeries(h1Buf, true);
   if(CopyBuffer(emaM15, 0, 0, 3, emaBuf) < 3) return;
   if(CopyBuffer(atrM15, 0, 0, 3, atrBuf) < 3) return;
   if(CopyBuffer(emaH1, 0, 0, 3, h1Buf) < 3) return;

   bool hasPos = false;
   if(PositionSelect(_Symbol) && (ulong)PositionGetInteger(POSITION_MAGIC) == MagicNumber)
      hasPos = true;

   if(hasPos)
   {
      double close1 = iClose(_Symbol, PERIOD_M15, 1);
      double distExit = MathAbs((close1 - emaBuf[1]) / atrBuf[1]);
      datetime openTime = (datetime)PositionGetInteger(POSITION_TIME);
      bool timeExit = (TimeCurrent() - openTime) >= 32 * PeriodSeconds(PERIOD_M15);
      if(distExit <= 0.25 || timeExit)
         trade.PositionClose(_Symbol);
   }

   datetime bar = iTime(_Symbol, PERIOD_M15, 0);
   if(bar == lastBar) return;
   lastBar = bar;

   if(dailyBlocked) return;
   if(tradesToday >= MaxTradesPerDay) return;
   if(hasPos) return;

   double closeM15 = iClose(_Symbol, PERIOD_M15, 1);
   double atr = atrBuf[1];
   if(atr <= 0.0 || closeM15 <= 0.0) return;
   double distance = (closeM15 - emaBuf[1]) / atr;
   if(distance > -2.25) return;
   if(h1Buf[1] <= h1Buf[2]) return;

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double stopDistance = atr * 2.0;
   double sl = NormalizeDouble(ask - stopDistance, _Digits);
   double tp = NormalizeDouble(ask + stopDistance * 2.0, _Digits);

   double riskMoney = AccountInfoDouble(ACCOUNT_EQUITY) * RiskPerTradePct / 100.0;
   double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double stepLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   if(tickSize <= 0.0 || tickValue <= 0.0 || stepLot <= 0.0) return;
   double lossOneLot = stopDistance / tickSize * tickValue;
   if(lossOneLot <= 0.0) return;
   double lots = riskMoney / lossOneLot;
   lots = MathFloor(lots / stepLot) * stepLot;
   if(lots < minLot) lots = minLot;
   if(lots > maxLot) lots = maxLot;
   lots = NormalizeDouble(lots, 2);

   if(trade.Buy(lots, _Symbol, 0.0, sl, tp))
      tradesToday++;
}
