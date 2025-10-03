# Task 13 Quick Start Guide

## Problem
All components exist but NOT connected via events.

## Solution
Connect event chain: MarketData → Candle → Analysis → Signal → Risk → Order

## New Events (events.py)
- CandleClosedEvent: symbol, interval, df, timestamp
- RiskApprovedOrderEvent: signal, approved_quantity, risk_params

## New Components
1. MarketDataAggregator: MarketDataEvent → CandleClosedEvent
2. StrategyCoordinator: CandleClosedEvent → SignalEvent

## Modifications
1. RiskManager: +SignalEvent handler → RiskApprovedOrderEvent
2. OrderExecutor: +RiskApprovedOrderEvent handler → auto-execute
3. SystemIntegrator: register new components

## Implementation Order
13.1 → 13.2 → 13.3 → 13.4 → 13.5 → 13.6 → 13.7 → 13.8

## Start Command
`/sc:implement task 13 --serena --sequential --task-manage`
