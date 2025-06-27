package com.example.intentive.data.model

enum class NotificationTiming(val displayName: String, val minutes: Long) {
    FIVE_MINUTES("5 minutes", 5),
    FIFTEEN_MINUTES("15 minutes", 15),
    THIRTY_MINUTES("30 minutes", 30),
    ONE_HOUR("1 hour", 60),
    TWO_HOURS("2 hours", 120),
    ONE_DAY("1 day", 1440)
}

data class NotificationSettings(
    val beforeEnabled: Boolean = false,
    val beforeTiming: NotificationTiming = NotificationTiming.FIFTEEN_MINUTES,
    val atTimeEnabled: Boolean = true,
    val afterEnabled: Boolean = false,
    val afterTiming: NotificationTiming = NotificationTiming.ONE_HOUR,
    val locationEnabled: Boolean = false
)
