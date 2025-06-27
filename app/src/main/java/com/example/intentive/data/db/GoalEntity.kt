package com.example.intentive.data.db

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "goals")
data class Goal(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    val behavior: String,
    val time: Long, // Store as milliseconds (timestamp) for easy scheduling and sorting
    val location: String, // Human-readable address
    val latitude: Double? = null, // GPS latitude
    val longitude: Double? = null, // GPS longitude
    // Notification settings
    val remindBefore: Boolean = false,
    val beforeTimingMinutes: Long = 15, // Default 15 minutes
    val notifyAtTime: Boolean = true,
    val promptReview: Boolean = false,
    val afterTimingMinutes: Long = 60, // Default 1 hour
    val notifyAtLocation: Boolean = false,
    // Status
    val isCompleted: Boolean = false,
    val reviewNotes: String? = null,
    val creationTimestamp: Long = System.currentTimeMillis()
)