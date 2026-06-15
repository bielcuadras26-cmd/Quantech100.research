//+------------------------------------------------------------------+
//| QuantTech100 Single Symbol Backtest EA                            |
//| M15 mean reversion with H1 trend filter                           |
//+------------------------------------------------------------------+
#property strict
#property version "1.00"

#include <Trade/Trade.mqh>

input ENUM_TIMEFRAMES SignalTF = PERIOD_M15;
input ENUM_TIMEFRAMES TrendTF = PERIOD_H1;
input int EmaPeriod = 50;
input int AtrPeriod = 14;
input double EntryDistanceAtr = 2.25;
input double ExitDistanceAtr = 0.25;
input double StopAtr = 2.0;
input double TakeProfitR = 2.0;
input int MaxBarsInTrade = 32;
input double RiskPerTradePct = 1.25;
input double DailyStopPct = 3.0;
input double OfficialDailyLimitPct = 5.0;
input int MaxTradesPerDay = 8;
input int SlippagePoints = 30;
input ulong MagicNumber = 10025001;

CTrade trade;

int ema_m15_handle = INVALID_HANDLE;
int atr_m15_handle = INVALID_HANDLE;
int ema_h1_handle = INVALID_HANDLE;

datetime last_bar_time = 0;
int current_day_key = 0;
double day_start_equity = 0.0;
bool daily_stopped = false;
int trades_today = 0;

//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints(SlippagePoints);

   ema_m15_handle = iMA(_Symbol, SignalTF, EmaPeriod, 0, MODE_EMA, PRICE_CLOSE);
   atr_m15_handle = iATR(_Symbol, SignalTF, AtrPeriod);
   ema_h1_handle = iMA(_Symbol, TrendTF, EmaPeriod, 0, MODE_EMA, PRICE_CLOSE);

   if(ema_m15_handle == INVALID_HANDLE || atr_m15_handle == INVALID_HANDLE || ema_h1_handle == INVALID_HANDLE)
   {
      Print("Error creating indicator handles");
      return INIT_FAILED;
   }

   RefreshDailyState();
   Print("QuantTech100 EA initialized on ", _Symbol);
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(ema_m15_handle != INVALID_HANDLE)
      IndicatorRelease(ema_m15_handle);
   if(atr_m15_handle != INVALID_HANDLE)
      IndicatorRelease(atr_m15_handle);
   if(ema_h1_handle != INVALID_HANDLE)
      IndicatorRelease(ema_h1_handle);
}

//+------------------------------------------------------------------+
void OnTick()
{
   RefreshDailyState();
   EnforceDailyRisk();
   ManageOpenPosition();

   if(daily_stopped)
      return;
   if(trades_today >= MaxTradesPerDay)
      return;
   if(!IsNewBar())
      return;
   if(HasOpenPosition())
      return;

   CheckEntry();
}

//+------------------------------------------------------------------+
void CheckEntry()
{
   double ema_m15 = GetBufferValue(ema_m15_handle, 1);
   double atr_m15 = GetBufferValue(atr_m15_handle, 1);
   double ema_h1_now = GetBufferValue(ema_h1_handle, 1);
   double ema_h1_prev = GetBufferValue(ema_h1_handle, 2);
   double close_m15 = iClose(_Symbol, SignalTF, 1);

   if(ema_m15 <= 0.0 || atr_m15 <= 0.0 || ema_h1_now <= 0.0 || ema_h1_prev <= 0.0 || close_m15 <= 0.0)
      return;

   double distance_atr = (close_m15 - ema_m15) / atr_m15;
   bool h1_slope_up = ema_h1_now > ema_h1_prev;

   if(distance_atr > -EntryDistanceAtr)
      return;
   if(!h1_slope_up)
      return;

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double stop_distance = atr_m15 * StopAtr;
   double sl = NormalizePrice(ask - stop_distance);
   double tp = NormalizePrice(ask + stop_distance * TakeProfitR);
   double volume = CalculateVolume(stop_distance);

   if(volume <= 0.0)
   {
      Print("Invalid volume");
      return;
   }

   bool ok = trade.Buy(volume, _Symbol, 0.0, sl, tp, "QT100 MR long");
   if(ok)
   {
      trades_today++;
      Print("BUY ", _Symbol, " volume=", DoubleToString(volume, 2), " sl=", DoubleToString(sl, _Digits), " tp=", DoubleToString(tp, _Digits));
   }
   else
   {
      Print("BUY failed. Retcode=", trade.ResultRetcode(), " ", trade.ResultRetcodeDescription());
   }
}

//+------------------------------------------------------------------+
void ManageOpenPosition()
{
   if(!SelectOurPosition())
      return;

   double ema_m15 = GetBufferValue(ema_m15_handle, 1);
   double atr_m15 = GetBufferValue(atr_m15_handle, 1);
   double close_m15 = iClose(_Symbol, SignalTF, 1);

   if(ema_m15 <= 0.0 || atr_m15 <= 0.0 || close_m15 <= 0.0)
      return;

   double distance_atr = MathAbs((close_m15 - ema_m15) / atr_m15);
   bool exit_signal = distance_atr <= ExitDistanceAtr;

   datetime open_time = (datetime)PositionGetInteger(POSITION_TIME);
   int max_seconds = MaxBarsInTrade * PeriodSeconds(SignalTF);
   bool time_exit = (TimeCurrent() - open_time) >= max_seconds;

   if(exit_signal || time_exit)
   {
      bool ok = trade.PositionClose(_Symbol);
      if(ok)
         Print("Closed ", _Symbol, " reason=", exit_signal ? "mean_exit" : "time_exit");
      else
         Print("Close failed. Retcode=", trade.ResultRetcode(), " ", trade.ResultRetcodeDescription());
   }
}

//+------------------------------------------------------------------+
void RefreshDailyState()
{
   MqlDateTime now;
   TimeToStruct(TimeCurrent(), now);
   int day_key = now.year * 10000 + now.mon * 100 + now.day;

   if(day_key != current_day_key)
   {
      current_day_key = day_key;
      day_start_equity = AccountInfoDouble(ACCOUNT_EQUITY);
      daily_stopped = false;
      trades_today = 0;
      Print("New trading day. Start equity=", DoubleToString(day_start_equity, 2));
   }
}

//+------------------------------------------------------------------+
void EnforceDailyRisk()
{
   if(day_start_equity <= 0.0)
      return;

   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double daily_return = (equity - day_start_equity) / day_start_equity;

   if(daily_return <= -OfficialDailyLimitPct / 100.0)
   {
      daily_stopped = true;
      CloseOurPosition();
      Print("Official daily limit protection triggered: ", DoubleToString(daily_return * 100.0, 2), "%");
      return;
   }

   if(daily_return <= -DailyStopPct / 100.0)
   {
      daily_stopped = true;
      CloseOurPosition();
      Print("Operational daily stop triggered: ", DoubleToString(daily_return * 100.0, 2), "%");
   }
}

//+------------------------------------------------------------------+
double CalculateVolume(double stop_distance_price)
{
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double risk_money = equity * RiskPerTradePct / 100.0;
   double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double max_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lot_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   if(risk_money <= 0.0 || stop_distance_price <= 0.0 || tick_size <= 0.0 || tick_value <= 0.0 || lot_step <= 0.0)
      return 0.0;

   double loss_per_lot = (stop_distance_price / tick_size) * tick_value;
   if(loss_per_lot <= 0.0)
      return 0.0;

   double lots = risk_money / loss_per_lot;
   lots = MathFloor(lots / lot_step) * lot_step;
   lots = MathMax(min_lot, MathMin(lots, max_lot));
   return NormalizeVolume(lots, lot_step);
}

//+------------------------------------------------------------------+
bool IsNewBar()
{
   datetime bar_time = iTime(_Symbol, SignalTF, 0);
   if(bar_time <= 0)
      return false;
   if(bar_time == last_bar_time)
      return false;

   last_bar_time = bar_time;
   return true;
}

//+------------------------------------------------------------------+
double GetBufferValue(int handle, int shift)
{
   double buffer[];
   ArraySetAsSeries(buffer, true);
   int copied = CopyBuffer(handle, 0, shift, 1, buffer);
   if(copied != 1)
      return 0.0;
   return buffer[0];
}

//+------------------------------------------------------------------+
bool HasOpenPosition()
{
   return SelectOurPosition();
}

//+------------------------------------------------------------------+
bool SelectOurPosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;
      if(!PositionSelectByTicket(ticket))
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != MagicNumber)
         continue;
      return true;
   }
   return false;
}

//+------------------------------------------------------------------+
void CloseOurPosition()
{
   if(SelectOurPosition())
      trade.PositionClose(_Symbol);
}

//+------------------------------------------------------------------+
double NormalizePrice(double price)
{
   return NormalizeDouble(price, _Digits);
}

//+------------------------------------------------------------------+
double NormalizeVolume(double lots, double step)
{
   int digits = 2;
   if(step == 0.1)
      digits = 1;
   else if(step == 0.01)
      digits = 2;
   else if(step == 0.001)
      digits = 3;
   else if(step == 0.0001)
      digits = 4;

   return NormalizeDouble(lots, digits);
}
