package com.example.intentive

import android.content.Context
import com.example.intentive.data.db.GoalDao
import com.example.intentive.data.db.IntentiveDatabase
import com.example.intentive.data.repository.GoalRepository
import com.example.intentive.services.NotificationScheduler
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object AppModule {

    @Singleton
    @Provides
    fun provideIntentiveDatabase(@ApplicationContext context: Context): IntentiveDatabase {
        return IntentiveDatabase.getDatabase(context)
    }

    @Singleton
    @Provides
    fun provideGoalDao(database: IntentiveDatabase): GoalDao {
        return database.goalDao()
    }

    @Singleton
    @Provides
    fun provideGoalRepository(goalDao: GoalDao): GoalRepository {
        return GoalRepository(goalDao)
    }

    @Singleton
    @Provides
    fun provideNotificationScheduler(@ApplicationContext context: Context): NotificationScheduler {
        return NotificationScheduler(context)
    }
}