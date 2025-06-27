package com.example.intentive.services

import android.content.Context
import android.util.Log
import androidx.work.*
import com.example.intentive.data.db.Goal
import com.example.intentive.services.workers.GoalNotificationWorker
import java.util.concurrent.TimeUnit
import javax.inject.Inject

class NotificationScheduler @Inject constructor(private val context: Context) {

    companion object {
        const val TAG = "NotificationScheduler"
        const val GOAL_ID_EXTRA = "goal_id_extra"
        const val NOTIFICATION_TYPE_EXTRA = "notification_type_extra"
        const val TYPE_PRE_GOAL = "pre_goal"
        const val TYPE_AT_TIME = "at_time"
        const val TYPE_POST_REVIEW = "post_review"

        // Define reminder intervals (can be configurable later)
        const val PRE_REMINDER_OFFSET_MINUTES = 15L // Default, now configurable per goal
        const val POST_REVIEW_OFFSET_HOURS = 1L     // Default, now configurable per goal
    }

    fun scheduleNotificationsForGoal(goal: Goal) {
        Log.d(TAG, "Scheduling notifications for goal ID: ${goal.id} at time: ${goal.time}")
        val currentTime = System.currentTimeMillis()
        Log.d(TAG, "Current time: $currentTime, Goal time: ${goal.time}, Difference: ${goal.time - currentTime}ms")

        if (goal.remindBefore) {
            val reminderTime = goal.time - TimeUnit.MINUTES.toMillis(goal.beforeTimingMinutes)
            Log.d(TAG, "Reminder time calculated: $reminderTime (${goal.beforeTimingMinutes} min before goal time)")
            if (reminderTime > currentTime) {
                scheduleWork(goal.id, reminderTime, TYPE_PRE_GOAL)
                Log.d(TAG, "Scheduled PRE_GOAL for ${goal.id} at $reminderTime (${goal.beforeTimingMinutes} min before)")
            } else {
                Log.d(TAG, "PRE_GOAL time for ${goal.id} is in the past. Not scheduling.")
            }
        }

        if (goal.notifyAtTime) {
             if (goal.time > currentTime) {
                scheduleWork(goal.id, goal.time, TYPE_AT_TIME)
                Log.d(TAG, "Scheduled AT_TIME for ${goal.id} at ${goal.time}")
            } else {
                Log.d(TAG, "AT_TIME for ${goal.id} is in the past. Not scheduling.")
            }
        }

        if (goal.promptReview) {
            val reviewTime = goal.time + TimeUnit.MINUTES.toMillis(goal.afterTimingMinutes)
            Log.d(TAG, "Review time calculated: $reviewTime (${goal.afterTimingMinutes} min after goal time)")
             if (reviewTime > currentTime) {
                scheduleWork(goal.id, reviewTime, TYPE_POST_REVIEW)
                Log.d(TAG, "Scheduled POST_REVIEW for ${goal.id} at $reviewTime (${goal.afterTimingMinutes} min after)")
            } else {
                Log.d(TAG, "POST_REVIEW time for ${goal.id} is in the past. Not scheduling.")
            }
        }
    }

    private fun scheduleWork(goalId: Int, timeMillis: Long, notificationType: String) {
        val delay = timeMillis - System.currentTimeMillis()
        if (delay <= 0) {
            Log.w(TAG, "Cannot schedule work for $notificationType (Goal ID: $goalId), time is in the past.")
            return
        }

        val data = workDataOf(
            GOAL_ID_EXTRA to goalId,
            NOTIFICATION_TYPE_EXTRA to notificationType
        )

        val workRequest = OneTimeWorkRequestBuilder<GoalNotificationWorker>()
            .setInitialDelay(delay, TimeUnit.MILLISECONDS)
            .setInputData(data)
            .addTag("goal_${goalId}_${notificationType}") // Unique tag for cancellation
            .build()

        WorkManager.getInstance(context).enqueueUniqueWork(
            "goal_notification_${goalId}_${notificationType}", // Unique work name
            ExistingWorkPolicy.REPLACE, // Replace if already exists
            workRequest
        )
        Log.i(TAG, "Enqueued WorkManager task for $notificationType, Goal ID: $goalId, Delay: $delay ms")
    }


    fun cancelNotificationsForGoal(goalId: Int) {
        // Cancel by tag for more specific cancellation
        WorkManager.getInstance(context).cancelAllWorkByTag("goal_${goalId}_$TYPE_PRE_GOAL")
        WorkManager.getInstance(context).cancelAllWorkByTag("goal_${goalId}_$TYPE_AT_TIME")
        WorkManager.getInstance(context).cancelAllWorkByTag("goal_${goalId}_$TYPE_POST_REVIEW")
        Log.i(TAG, "Cancelled all notifications for Goal ID: $goalId")
    }
    
    // Test notification function for debugging
    fun sendTestNotification() {
        Log.d(TAG, "Scheduling test notification in 5 seconds...")
        
        val testGoal = Goal(
            id = 999,
            behavior = "Test Notification",
            time = System.currentTimeMillis(),
            location = "Test Location",
            notifyAtTime = true
        )
        
        // Also create the test goal temporarily for the worker
        val data = workDataOf(
            GOAL_ID_EXTRA to 999,
            NOTIFICATION_TYPE_EXTRA to TYPE_AT_TIME,
            "test_goal_behavior" to testGoal.behavior,
            "test_goal_location" to testGoal.location,
            "test_goal_time" to testGoal.time
        )
        
        val workRequest = OneTimeWorkRequestBuilder<GoalNotificationWorker>()
            .setInitialDelay(5, TimeUnit.SECONDS)
            .setInputData(data)
            .addTag("test_notification")
            .setConstraints(
                Constraints.Builder()
                    .setRequiredNetworkType(NetworkType.NOT_REQUIRED)
                    .setRequiresBatteryNotLow(false)
                    .setRequiresCharging(false)
                    .setRequiresDeviceIdle(false)
                    .setRequiresStorageNotLow(false)
                    .build()
            )
            .build()
            
        WorkManager.getInstance(context).enqueueUniqueWork(
            "test_notification",
            ExistingWorkPolicy.REPLACE,
            workRequest
        )
        
        Log.i(TAG, "Test notification scheduled!")
    }
    
    // Immediate test notification (3 seconds)
    fun sendImmediateTestNotification() {
        Log.d(TAG, "Scheduling immediate test notification in 3 seconds...")
        
        val testGoal = Goal(
            id = 998,
            behavior = "Immediate Test",
            time = System.currentTimeMillis(),
            location = "Test Location",
            notifyAtTime = true
        )
        
        // Schedule a test notification for 3 seconds from now
        val testTime = System.currentTimeMillis() + 3000
        scheduleWork(998, testTime, TYPE_AT_TIME)
        
        Log.i(TAG, "Immediate test notification scheduled!")
    }
    
    // Direct notification test (bypasses WorkManager)
    fun sendDirectNotification() {
        Log.d(TAG, "Sending direct notification immediately...")
        
        try {
            val goalNotificationWorker = GoalNotificationWorkerHelper(context)
            goalNotificationWorker.sendDirectTestNotification()
            Log.i(TAG, "Direct notification sent!")
        } catch (e: Exception) {
            Log.e(TAG, "Error sending direct notification: ${e.message}", e)
        }
    }
    
    // Method to check scheduled work
    fun checkScheduledWork() {
        val workManager = WorkManager.getInstance(context)
        workManager.getWorkInfosByTag("goal_").let { workInfos ->
            Log.d(TAG, "Checking scheduled work...")
            for (workInfo in workInfos.get()) {
                Log.d(TAG, "Work: ${workInfo.id}, State: ${workInfo.state}, Tags: ${workInfo.tags}")
            }
        }
    }
}