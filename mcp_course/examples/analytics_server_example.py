#!/usr/bin/env python3
"""
Analytics Server Example for Multi-Server Orchestration

This server demonstrates analytics and statistical analysis capabilities
for use in multi-server workflows.
"""

import asyncio
from datetime import datetime, timedelta
import json
import statistics
from typing import Any

from mcp import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from mcp_course.server.scaffolding import ServerConfig


class AnalyticsServer:
    """
    MCP Server that provides analytics and statistical analysis tools.

    This server demonstrates:
    - Statistical calculations and metrics
    - Trend analysis and forecasting
    - Data visualization recommendations
    - Performance analytics
    """

    def __init__(self, config: ServerConfig):
        """Initialize the analytics server."""
        self.config = config
        self.server = Server(config.name)
        self._setup_handlers()

    def _setup_handlers(self):
        """Set up MCP protocol handlers."""

        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """List available analytics tools."""
            return [
                Tool(
                    name="calculate_statistics",
                    description="Calculate statistical metrics for numerical data",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "string",
                                "description": "JSON string containing data for analysis"
                            },
                            "metrics": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["mean", "median", "mode", "std", "var", "min", "max", "range", "quartiles", "skewness"]
                                },
                                "description": "Statistical metrics to calculate",
                                "default": ["mean", "median", "std"]
                            },
                            "group_by": {
                                "type": "string",
                                "description": "Field to group statistics by"
                            },
                            "numerical_fields": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Specific numerical fields to analyze"
                            }
                        },
                        "required": ["data"]
                    }
                ),

                Tool(
                    name="analyze_trends",
                    description="Analyze trends and patterns in time-series data",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "string",
                                "description": "JSON string containing time-series data"
                            },
                            "period": {
                                "type": "string",
                                "enum": ["daily", "weekly", "monthly", "quarterly"],
                                "description": "Time period for trend analysis",
                                "default": "daily"
                            },
                            "date_field": {
                                "type": "string",
                                "description": "Field containing date/time information",
                                "default": "date"
                            },
                            "value_field": {
                                "type": "string",
                                "description": "Field containing values to analyze",
                                "default": "amount"
                            },
                            "forecast_periods": {
                                "type": "integer",
                                "description": "Number of periods to forecast",
                                "default": 0
                            }
                        },
                        "required": ["data"]
                    }
                ),

                Tool(
                    name="correlation_analysis",
                    description="Analyze correlations between different variables",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "string",
                                "description": "JSON string containing data for correlation analysis"
                            },
                            "variables": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Variables to analyze for correlations"
                            },
                            "method": {
                                "type": "string",
                                "enum": ["pearson", "spearman", "kendall"],
                                "description": "Correlation method to use",
                                "default": "pearson"
                            }
                        },
                        "required": ["data"]
                    }
                ),

                Tool(
                    name="performance_metrics",
                    description="Calculate performance and business metrics",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "string",
                                "description": "JSON string containing performance data"
                            },
                            "metric_type": {
                                "type": "string",
                                "enum": ["sales", "customer", "operational", "financial"],
                                "description": "Type of performance metrics to calculate"
                            },
                            "time_period": {
                                "type": "string",
                                "description": "Time period for performance calculation"
                            },
                            "benchmark_data": {
                                "type": "string",
                                "description": "Benchmark data for comparison"
                            }
                        },
                        "required": ["data", "metric_type"]
                    }
                )
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            """Handle tool execution."""

            if name == "calculate_statistics":
                return await self._calculate_statistics(arguments)
            elif name == "analyze_trends":
                return await self._analyze_trends(arguments)
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

    async def _calculate_statistics(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Calculate statistical metrics for numerical data."""
        try:
            # Parse input data
            data_str = arguments["data"]
            data_obj = json.loads(data_str) if isinstance(data_str, str) else data_str

            # Extract the actual data array
            data_records = data_obj["data"] if isinstance(data_obj, dict) and "data" in data_obj else data_obj

            metrics = arguments.get("metrics", ["mean", "median", "std"])
            group_by = arguments.get("group_by")
            numerical_fields = arguments.get("numerical_fields")

            # Identify numerical fields if not specified
            if not numerical_fields and data_records:
                numerical_fields = []
                sample_record = data_records[0]
                for key, value in sample_record.items():
                    if isinstance(value, (int, float)):
                        numerical_fields.append(key)

            result = {
                "analysis_timestamp": datetime.now().isoformat(),
                "total_records": len(data_records),
                "numerical_fields": numerical_fields,
                "metrics_calculated": metrics,
                "statistics": {}
            }

            if group_by:
                # Group-wise statistics
                groups = {}
                for record in data_records:
                    group_key = record.get(group_by, "unknown")
                    if group_key not in groups:
                        groups[group_key] = []
                    groups[group_key].append(record)

                for group_key, group_records in groups.items():
                    group_stats = {}
                    for field in numerical_fields:
                        values = [r.get(field, 0) for r in group_records if isinstance(r.get(field), (int, float))]
                        if values:
                            group_stats[field] = self._calculate_field_statistics(values, metrics)
                    result["statistics"][group_key] = group_stats
            else:
                # Overall statistics
                for field in numerical_fields:
                    values = [r.get(field, 0) for r in data_records if isinstance(r.get(field), (int, float))]
                    if values:
                        result["statistics"][field] = self._calculate_field_statistics(values, metrics)

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(type="text", text=f"Error calculating statistics: {e!s}")]

    async def _analyze_trends(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Analyze trends and patterns in time-series data."""
        try:
            # Parse input data
            data_str = arguments["data"]
            data_obj = json.loads(data_str) if isinstance(data_str, str) else data_str

            # Extract the actual data array
            data_records = data_obj["data"] if isinstance(data_obj, dict) and "data" in data_obj else  data_obj

            period = arguments.get("period", "daily")
            date_field = arguments.get("date_field", "date")
            value_field = arguments.get("value_field", "amount")
            forecast_periods = arguments.get("forecast_periods", 0)

            # Group data by time period
            time_series = self._group_by_time_period(data_records, date_field, value_field, period)

            # Calculate trend metrics
            trend_analysis = self._calculate_trend_metrics(time_series)

            result = {
                "analysis_timestamp": datetime.now().isoformat(),
                "period": period,
                "date_field": date_field,
                "value_field": value_field,
                "data_points": len(time_series),
                "trend_analysis": trend_analysis,
                "time_series_data": time_series
            }

            # Add forecasting if requested
            if forecast_periods > 0:
                forecast = self._simple_forecast(time_series, forecast_periods)
                result["forecast"] = forecast

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(type="text", text=f"Error analyzing trends: {e!s}")]





    def _calculate_field_statistics(self, values: list[float], metrics: list[str]) -> dict[str, float]:
        """Calculate statistics for a field."""
        stats = {}

        if "mean" in metrics:
            stats["mean"] = statistics.mean(values)
        if "median" in metrics:
            stats["median"] = statistics.median(values)
        if "mode" in metrics and len(values) > 1:
            try:
                stats["mode"] = statistics.mode(values)
            except statistics.StatisticsError:
                stats["mode"] = None  # No unique mode
        if "std" in metrics and len(values) > 1:
            stats["std"] = statistics.stdev(values)
        if "var" in metrics and len(values) > 1:
            stats["var"] = statistics.variance(values)
        if "min" in metrics:
            stats["min"] = min(values)
        if "max" in metrics:
            stats["max"] = max(values)
        if "range" in metrics:
            stats["range"] = max(values) - min(values)
        if "quartiles" in metrics and len(values) >= 4:
            stats["quartiles"] = statistics.quantiles(values, n=4)

        return stats

    def _group_by_time_period(self, data_records: list[dict[str, Any]], date_field: str, value_field: str, period: str) -> list[dict[str, Any]]:
        """Group data by time period."""
        time_groups = {}

        for record in data_records:
            date_str = record.get(date_field)
            value = record.get(value_field, 0)

            if not date_str or not isinstance(value, (int, float)):
                continue

            try:
                # Parse date
                if isinstance(date_str, str):
                    date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                else:
                    date_obj = date_str

                # Group by period
                if period == "daily":
                    period_key = date_obj.strftime("%Y-%m-%d")
                elif period == "weekly":
                    # Get Monday of the week
                    monday = date_obj - timedelta(days=date_obj.weekday())
                    period_key = monday.strftime("%Y-%m-%d")
                elif period == "monthly":
                    period_key = date_obj.strftime("%Y-%m")
                elif period == "quarterly":
                    quarter = (date_obj.month - 1) // 3 + 1
                    period_key = f"{date_obj.year}-Q{quarter}"
                else:
                    period_key = date_obj.strftime("%Y-%m-%d")

                if period_key not in time_groups:
                    time_groups[period_key] = []
                time_groups[period_key].append(value)

            except (ValueError, AttributeError):
                continue

        # Convert to time series format
        time_series = []
        for period_key in sorted(time_groups.keys()):
            values = time_groups[period_key]
            time_series.append({
                "period": period_key,
                "value": sum(values),
                "count": len(values),
                "average": statistics.mean(values) if values else 0
            })

        return time_series

    def _calculate_trend_metrics(self, time_series: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate trend metrics from time series data."""
        if len(time_series) < 2:
            return {"error": "Insufficient data for trend analysis"}

        values = [point["value"] for point in time_series]

        # Calculate basic trend metrics
        trend_metrics = {
            "total_periods": len(time_series),
            "total_value": sum(values),
            "average_per_period": statistics.mean(values),
            "growth_rate": self._calculate_growth_rate(values),
            "volatility": statistics.stdev(values) if len(values) > 1 else 0,
            "trend_direction": self._determine_trend_direction(values)
        }

        # Calculate period-over-period changes
        changes = []
        for i in range(1, len(values)):
            if values[i-1] != 0:
                change = ((values[i] - values[i-1]) / values[i-1]) * 100
                changes.append(change)

        if changes:
            trend_metrics["average_change_percent"] = statistics.mean(changes)
            trend_metrics["max_increase"] = max(changes)
            trend_metrics["max_decrease"] = min(changes)

        return trend_metrics

    def _calculate_growth_rate(self, values: list[float]) -> float:
        """Calculate overall growth rate."""
        if len(values) < 2 or values[0] == 0:
            return 0.0

        return ((values[-1] - values[0]) / values[0]) * 100

    def _determine_trend_direction(self, values: list[float]) -> str:
        """Determine overall trend direction."""
        if len(values) < 2:
            return "insufficient_data"

        increases = 0
        decreases = 0

        for i in range(1, len(values)):
            if values[i] > values[i-1]:
                increases += 1
            elif values[i] < values[i-1]:
                decreases += 1

        if increases > decreases:
            return "increasing"
        elif decreases > increases:
            return "decreasing"
        else:
            return "stable"

    def _simple_forecast(self, time_series: list[dict[str, Any]], periods: int) -> list[dict[str, Any]]:
        """Simple linear forecast based on trend."""
        if len(time_series) < 2:
            return []

        values = [point["value"] for point in time_series]

        # Simple linear trend calculation
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))

        # Linear regression coefficients
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        intercept = (y_sum - slope * x_sum) / n

        # Generate forecast
        forecast = []
        for i in range(periods):
            future_x = n + i
            predicted_value = slope * future_x + intercept
            forecast.append({
                "period": f"forecast_{i+1}",
                "predicted_value": max(0, predicted_value),  # Ensure non-negative
                "confidence": "low"  # Simple forecast has low confidence
            })

        return forecast

    def _calculate_correlation(self, values1: list[float], values2: list[float], method: str) -> float:
        """Calculate correlation between two variables."""
        if len(values1) != len(values2) or len(values1) < 2:
            return 0.0

        # Simple Pearson correlation implementation
        if method == "pearson":
            mean1 = statistics.mean(values1)
            mean2 = statistics.mean(values2)

            numerator = sum((x - mean1) * (y - mean2) for x, y in zip(values1, values2, strict=False))

            sum_sq1 = sum((x - mean1) ** 2 for x in values1)
            sum_sq2 = sum((y - mean2) ** 2 for y in values2)

            denominator = (sum_sq1 * sum_sq2) ** 0.5

            if denominator == 0:
                return 0.0

            return numerator / denominator

        # For other methods, return a placeholder
        return 0.0

    def _interpret_correlation_strength(self, correlation: float) -> str:
        """Interpret correlation strength."""
        abs_corr = abs(correlation)

        if abs_corr >= 0.8:
            return "very_strong"
        elif abs_corr >= 0.6:
            return "strong"
        elif abs_corr >= 0.4:
            return "moderate"
        elif abs_corr >= 0.2:
            return "weak"
        else:
            return "very_weak"

    def _generate_correlation_interpretation(self, correlations: dict[str, Any]) -> list[str]:
        """Generate interpretation of correlation results."""
        interpretations = []

        for pair, data in correlations.items():
            correlation = data["correlation"]
            strength = data["strength"]

            if abs(correlation) >= 0.6:
                direction = "positive" if correlation > 0 else "negative"
                interpretations.append(f"{pair} shows a {strength} {direction} correlation ({correlation:.3f})")

        if not interpretations:
            interpretations.append("No strong correlations found between the analyzed variables")

        return interpretations



    def _generate_performance_recommendations(self, metrics: dict[str, Any], metric_type: str) -> list[str]:
        """Generate performance recommendations based on metrics."""
        recommendations = []

        if metric_type == "sales":
            if metrics.get("average_transaction_value", 0) < 50:
                recommendations.append("Consider strategies to increase average transaction value")
            if metrics.get("total_transactions", 0) < 100:
                recommendations.append("Focus on increasing transaction volume")

        elif metric_type == "customer":
            if metrics.get("average_interactions_per_customer", 0) < 2:
                recommendations.append("Improve customer engagement and retention")

        if not recommendations:
            recommendations.append("Performance metrics are within expected ranges")

        return recommendations

    async def run(self):
        """Run the analytics server."""
        async with stdio_server() as (read_stream, write_stream):
            init_options = InitializationOptions(
                server_name=self.config.name,
                server_version=self.config.version,
                capabilities=self.config.capabilities
            )
            await self.server.run(read_stream, write_stream, init_options)


async def main():
    """Main entry point."""
    config = ServerConfig(
        name="analytics-server",
        version="1.0.0",
        description="Analytics and statistical analysis server for multi-server workflows"
    )

    server = AnalyticsServer(config)
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
