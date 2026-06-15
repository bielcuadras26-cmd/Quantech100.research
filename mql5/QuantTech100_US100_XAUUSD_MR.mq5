//+------------------------------------------------------------------+
//| QuantTech100 US100 + XAUUSD Mean Reversion EA                    |
//| Research version. Test in demo/strategy tester before live use.  |
//+------------------------------------------------------------------+
#property strict
#property version   "0.10"

#include <Trade/Trade.mqh>

input string          InpSymbols              = "US100,XAUUSD";
input ENUM_TIMEFRAMES InpSignalTimeframe      = PERIOD_M15;
input ENUM_TIMEFRAMES InpTrendTimeframe       = PERIOD_H1;
input int             InpEmaPeriod            = 50;
input int             InpAtrPeriod            = 14;
input double          InpEntryDistanceAtr     = 2.25;
input double          InpExitDistanceAtr      = 0.25;
input double          InpStopAtr              = 2.0;
input double          InpTakeProfitR          = 2.0;
input int             InpMaxBarsInTrade       = 32;
input double          InpRiskPerTradePct      = 1.25;
input double          InpDailyStopPct         = 3.0;
input double          InpOfficialDailyLimitPct= 5.0;
input int             InpMaxTradesPerDay      = 8;
input int             InpSlippagePoints       = 30;
input ulong           InpMagic                = 10025001;

struct SymbolState
{
   string   symbol;
   int      ema_m15;
   int      atr_m15;
   int      ema_h1;
   datetime last_bar_time;
};

CTrade trade;
SymbolState states[];

string daily_prefix;
int current_day_key = 0;
double day_start_equity = 0.0;
bool daily_stopped = false;
int trades_today = 0;

//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(InpSlippagePoints);
   daily_prefix = StringFormat("QT100_%I64u_%I64u_", AccountInfoInteger(ACCOUNT_LOGIN), InpMagic);

   string symbols[];
   int count = StringSplit(InpSymbols, ',', symbols);
   if(count <= 0)
   {
      Print("No symbols configured.");
      return INIT_FAILED;
   }

   ArrayResize(states, count);
   for(int i = 0; i < count; i++)
   {
      string sym = Trim(symbols[i]);
      if(sym == "")
         continue;

      if(!SymbolSelect(sym, true))
      {
         Print("Cannot select symbol: ", sym);
         return INIT_FAILED;
      }

      states[i].symbol = sym;
      states[i].ema_m15 = iMA(sym, InpSignalTimeframe, InpEmaPeriod, 0, MODE_EMA, PRICE_CLOSE);
      states[i].atr_m15 = iATR(sym, InpSignalTimeframe, InpAtrPeriod);
      states[i].ema_h1 = iMA(sym, InpTrendTimeframe, InpEmaPeriod, 0, MODE_EMA, PRICE_CLOSE);
      states[i].last_bar_time = 0;

      if(states[i].ema_m15 == INVALID_HANDLE || states[i].atr_m15 == INVALID_HANDLE || states[i].ema_h1 == INVALID_HANDLE)
      {
         Print("Cannot create indicator handles for: ", sym);
         return INIT_FAILED;
      }
   }

   RefreshDailyState();
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   for(int i = 0; i < ArraySize(states); i++)
   {
      if(states[i].ema_m15 != INVALID_HANDLE) IndicatorRelease(states[i].ema_m15);
      if(states[i].atr_m15 != INVALID_HANDLE) IndicatorRelease(states[i].atr_m15);
      if(states[i].ema_h1 != INVALID_HANDLE) IndicatorRelease(states[i].ema_h1);
   }
}

//+------------------------------------------------------------------+
void OnTick()
{
   RefreshDailyState();
   EnforceDailyRisk();

   for(int i = 0; i < ArraySize(states); i++)
   {
      if(states[i].symbol == "")
         continue;

      ManageOpenPosition(states[i]);

      if(daily_stopped)
         continue;
      if(trades_today >= InpMaxTradesPerDay)
         continue;
      if(!IsNewSignalBar(states[i]))
         continue;

      CheckEntry(states[i]);
   }
}

//+------------------------------------------------------------------+
void CheckEntry(SymbolState &state)
{
   string sym = state.symbol;
   if(HasOurPosition(sym))
      return;

   double ema = BufferValue(state.ema_m15, 1);
   double atr = BufferValue(state.atr_m15, 1);
   double h1_now = BufferValue(state.ema_h1, 1);
   double h1_prev = BufferValue(state.ema_h1, 2);
   double close_price = iClose(sym, InpSignalTimeframe, 1);

   if(ema <= 0 || atr <= 0 || h1_now <= 0 || h1_prev <= 0 || close_price <= 0)
      return;

   double distance_atr = (close_price - ema) / atr;
   bool h1_slope_up = (h1_now - h1_prev) > 0.0;

   if(distance_atr > -InpEntryDistanceAtr)
      return;
   if(!h1_slope_up)
      return;

   double stop_distance = atr * InpStopAtr;
   double ask = SymbolInfoDouble(sym, SYMBOL_ASK);
   double sl = NormalizePrice(sym, ask - stop_distance);
   double tp = NormalizePrice(sym, ask + stop_distance * InpTakeProfitR);
   double volume = CalculateVolume(sym, stop_distance);

   if(volume <= 0)
   {
      Print("Invalid volume for ", sym);
      return;
   }

   if(trade.Buy(volume, sym, 0.0, sl, tp, "QT100 mean reversion long"))
   {
      trades_today++;
      SaveTradesToday();
      Print("BUY ", sym, " volume=", volume, " sl=", sl, " tp=", tp);
   }
   else
   {
      Print("Buy failed ", sym, " retcode=", trade.ResultRetcode(), " ", trade.ResultRetcodeDescription());
   }
}

//+------------------------------------------------------------------+
void ManageOpenPosition(SymbolState &state)
{
   string sym = state.symbol;
   if(!SelectOurPosition(sym))
      return;

   double ema = BufferValue(state.ema_m15, 1);
   double atr = BufferValue(state.atr_m15, 1);
   double close_price = iClose(sym, InpSignalTimeframe, 1);
   if(ema <= 0 || atr <= 0 || close_price <= 0)
      return;

   double distance_atr = MathAbs((close_price - ema) / atr);
   bool exit_signal = distance_atr <= InpExitDistanceAtr;

   datetime open_time = (datetime)PositionGetInteger(POSITION_TIME);
   int max_seconds = InpMaxBarsInTrade * PeriodSeconds(InpSignalTimeframe);
   bool time_exit = (TimeCurrent() - open_time) >= max_seconds;

   if(exit_signal || time_exit)
   {
      if(trade.PositionClose(sym))
         Print("Closed ", sym, " reason=", exit_signal ? "mean_exit" : "time_exit");
      else
         Print("Close failed ", sym, " retcode=", trade.ResultRetcode(), " ", trade.ResultRetcodeDescription());
   }
}

//+------------------------------------------------------------------+
void EnforceDailyRisk()
{
   if(day_start_equity <= 0.0)
      return;

   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double daily_return = (equity - day_start_equity) / day_start_equity;

   if(daily_return <= -InpOfficialDailyLimitPct / 100.0)
   {
      daily_stopped = true;
      CloseAllOurPositions("official daily limit protection");
      SaveDailyStopped();
      return;
   }

   if(daily_return <= -InpDailyStopPct / 100.0)
   {
      daily_stopped = true;
      CloseAllOurPositions("operational daily stop");
      SaveDailyStopped();
   }
}

//+------------------------------------------------------------------+
void CloseAllOurPositions(string reason)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;
      if(!PositionSelectByTicket(ticket))
         continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != InpMagic)
         continue;

      string sym = PositionGetString(POSITION_SYMBOL);
      if(trade.PositionClose(sym))
         Print("Daily stop closed ", sym, ": ", reason);
   }
}

//+------------------------------------------------------------------+
bool IsNewSignalBar(SymbolState &state)
{
   datetime bar_time = iTime(state.symbol, InpSignalTimeframe, 0);
   if(bar_time <= 0)
      return false;
   if(bar_time == state.last_bar_time)
      return false;
   state.last_bar_time = bar_time;
   return true;
}

//+------------------------------------------------------------------+
double CalculateVolume(string sym, double stop_distance_price)
{
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double risk_money = equity * InpRiskPerTradePct / 100.0;
   double tick_size = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_SIZE);
   double tick_value = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_VALUE);
   double volume_min = SymbolInfoDouble(sym, SYMBOL_VOLUME_MIN);
   double volume_max = SymbolInfoDouble(sym, SYMBOL_VOLUME_MAX);
   double volume_step = SymbolInfoDouble(sym, SYMBOL_VOLUME_STEP);

   if(risk_money <= 0 || stop_distance_price <= 0 || tick_size <= 0 || tick_value <= 0 || volume_step <= 0)
      return 0.0;

   double loss_per_lot = (stop_distance_price / tick_size) * tick_value;
   if(loss_per_lot <= 0)
      return 0.0;

   double volume = risk_money / loss_per_lot;
   volume = MathFloor(volume / volume_step) * volume_step;
   volume = MathMax(volume_min, MathMin(volume, volume_max));
   return NormalizeVolume(sym, volume);
}

//+------------------------------------------------------------------+
bool HasOurPosition(string sym)
{
   return SelectOurPosition(sym);
}

//+------------------------------------------------------------------+
bool SelectOurPosition(string sym)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;
      if(!PositionSelectByTicket(ticket))
         continue;
      if(PositionGetString(POSITION_SYMBOL) != sym)
         continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != InpMagic)
         continue;
      return true;
   }
   return false;
}

//+------------------------------------------------------------------+
double BufferValue(int handle, int shift)
{
   double values[];
   ArraySetAsSeries(values, true);
   if(CopyBuffer(handle, 0, shift, 1, values) != 1)
      return 0.0;
   return values[0];
}

//+------------------------------------------------------------------+
double NormalizePrice(string sym, double price)
{
   int digits = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);
   return NormalizeDouble(price, digits);
}

//+------------------------------------------------------------------+
double NormalizeVolume(string sym, double volume)
{
   double step = SymbolInfoDouble(sym, SYMBOL_VOLUME_STEP);
   int digits = 2;
   if(step > 0)
   {
      digits = (int)MathRound(-MathLog10(step));
      if(digits < 0) digits = 0;
      if(digits > 8) digits = 8;
   }
   return NormalizeDouble(volume, digits);
}

//+------------------------------------------------------------------+
void RefreshDailyState()
{
   MqlDateTime now;
   TimeToStruct(TimeCurrent(), now);
   int day_key = now.year * 10000 + now.mon * 100 + now.day;
   if(day_key == current_day_key)
      return;

   current_day_key = day_key;
   string equity_key = daily_prefix + "day_start_equity";
   string day_key_name = daily_prefix + "day_key";
   string stopped_key = daily_prefix + "daily_stopped";
   string trades_key = daily_prefix + "trades_today";

   bool has_existing_day = GlobalVariableCheck(day_key_name) && (int)GlobalVariableGet(day_key_name) == day_key;
   if(!has_existing_day)
   {
      GlobalVariableSet(day_key_name, (double)day_key);
      GlobalVariableSet(equity_key, AccountInfoDouble(ACCOUNT_EQUITY));
      GlobalVariableSet(stopped_key, 0.0);
      GlobalVariableSet(trades_key, 0.0);
   }

   day_start_equity = GlobalVariableGet(equity_key);
   daily_stopped = GlobalVariableCheck(stopped_key) && GlobalVariableGet(stopped_key) > 0.5;
   trades_today = GlobalVariableCheck(trades_key) ? (int)GlobalVariableGet(trades_key) : 0;
}

//+------------------------------------------------------------------+
void SaveDailyStopped()
{
   GlobalVariableSet(daily_prefix + "daily_stopped", daily_stopped ? 1.0 : 0.0);
}

//+------------------------------------------------------------------+
void SaveTradesToday()
{
   GlobalVariableSet(daily_prefix + "trades_today", (double)trades_today);
}

//+------------------------------------------------------------------+
string Trim(string value)
{
   StringTrimLeft(value);
   StringTrimRight(value);
   return value;
}
