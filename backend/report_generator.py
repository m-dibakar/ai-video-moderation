def generate_report(video_path, violations, duration, model_version, loss_function):
    total_violation_time = sum(v.end_time - v.start_time for v in violations)

    return {
        "video": video_path,
        "duration_seconds": round(duration, 2),
        "model_version": model_version,
        "loss_function_used": loss_function,
        "summary": {
            "total_violations": len(violations),
            "violation_time_seconds": round(total_violation_time, 2),
            "violation_percentage": round(total_violation_time / max(duration, 1) * 100, 1),
            "compliance_status": "FAIL" if violations else "PASS",
            "severity_breakdown": {
                "high":   sum(1 for v in violations if v.severity == "high"),
                "medium": sum(1 for v in violations if v.severity == "medium"),
                "low":    sum(1 for v in violations if v.severity == "low"),
            }
        },
        "violations": [
            {
                "id": i + 1,
                "start_time": v.start_time,
                "end_time": v.end_time,
                "duration": round(v.end_time - v.start_time, 2),
                "timestamp_display": _fmt(v.start_time),
                "type": v.violation_type,
                "severity": v.severity,
                "confidence": v.avg_confidence,
                "suggested_action": v.suggested_action
            }
            for i, v in enumerate(violations)
        ]
    }

def _fmt(seconds):
    m, s = int(seconds // 60), int(seconds % 60)
    return f"{m:02d}:{s:02d}"
