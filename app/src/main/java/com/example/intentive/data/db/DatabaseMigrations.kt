package com.example.intentive.data.db

import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase

val MIGRATION_1_2 = object : Migration(1, 2) {
    override fun migrate(database: SupportSQLiteDatabase) {
        // Add latitude and longitude columns to goals table
        database.execSQL("ALTER TABLE goals ADD COLUMN latitude REAL")
        database.execSQL("ALTER TABLE goals ADD COLUMN longitude REAL")
    }
}

val MIGRATION_2_3 = object : Migration(2, 3) {
    override fun migrate(database: SupportSQLiteDatabase) {
        // Add new notification timing columns
        database.execSQL("ALTER TABLE goals ADD COLUMN beforeTimingMinutes INTEGER NOT NULL DEFAULT 15")
        database.execSQL("ALTER TABLE goals ADD COLUMN afterTimingMinutes INTEGER NOT NULL DEFAULT 60")
        database.execSQL("ALTER TABLE goals ADD COLUMN notifyAtLocation INTEGER NOT NULL DEFAULT 0")
        
        // Update existing notification settings to match new structure
        database.execSQL("UPDATE goals SET remindBefore = 0 WHERE remindBefore IS NULL")
        database.execSQL("UPDATE goals SET notifyAtTime = 1 WHERE notifyAtTime IS NULL")
        database.execSQL("UPDATE goals SET promptReview = 0 WHERE promptReview IS NULL")
    }
}
