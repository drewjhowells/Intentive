package com.example.intentive.services.workers

import android.Manifest
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.util.Log
import androidx.core.app.ActivityCompat
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.hilt.work.HiltWorker
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import com.example.intentive.MainActivity // Assuming MainActivity is your entry point
import com.example.intentive.R
import com.example.intentive.data.db.Goal
import com.example.intentive.data.repository.GoalRepository
import com.example.intentive.services.NotificationScheduler
import dagger.assisted.Assisted
import dagger.assisted.AssistedInject
import java.text.SimpleDateFormat
import java.util.*

@HiltWorker
class GoalNotificationWorker @AssistedInject constructor(
    @Assisted appContext: Context,
    @Assisted workerParams: WorkerParameters,
    private val goalRepository: GoalRepository // Hilt will inject this
) : CoroutineWorker(appContext, workerParams) {

    companion object {
        const val CHANNEL_ID = "intentive_goal_channel"
        const val CHANNEL_NAME = "Intentive Goals"
        const val TAG = "GoalNotificationWorker"
    }

    override suspend fun doWork(): Result {
        val goalId = inputData.getInt(NotificationScheduler.GOAL_ID_EXTRA, -1)
        val notificationType = inputData.getString(NotificationScheduler.NOTIFICATION_TYPE_EXTRA)

        if (goalId == -1 || notificationType == null) {
            Log.e(TAG, "Goal ID or Notification Type missing in worker data.")
            return Result.failure()
        }

        Log.d(TAG, "Worker started for Goal ID: $goalId, Type: $notificationType")

        val goal = goalRepository.getGoalByIdSync(goalId)
        if (goal == null) {
            // Check if this is a test notification
            if (goalId == 999 || goalId == 998) {
                Log.d(TAG, "Processing test notification for ID: $goalId")
                val testBehavior = inputData.getString("test_goal_behavior") ?: "Test Notification"
                val testLocation = inputData.getString("test_goal_location") ?: "Test Location"
                val testTime = inputData.getLong("test_goal_time", System.currentTimeMillis())
                
                val testGoal = Goal(
                    id = goalId,
                    behavior = testBehavior,
                    time = testTime,
                    location = testLocation,
                    notifyAtTime = true
                )
                
                createNotificationChannel()
                sendNotification(testGoal, notificationType)
                return Result.success()
            }
            
            Log.e(TAG, "Goal not found for ID: $goalId. Cannot send notification.")
            return Result.failure() // Or success if goal was deleted and that's okay
        }

        // Don't show notifications for completed goals unless it's a review prompt
        if (goal.isCompleted && notificationType != NotificationScheduler.TYPE_POST_REVIEW) {
            Log.i(TAG, "Goal ID: $goalId is already completed. Skipping $notificationType notification.")
            return Result.success()
        }


        createNotificationChannel()
        sendNotification(goal, notificationType)

        return Result.success()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                CHANNEL_NAME,
                NotificationManager.IMPORTANCE_HIGH
            ).apply {
                description = "Notifications for your Intentive goals"
                // enableLights(true)
                // lightColor = Color.BLUE
                // enableVibration(true)
            }
            val notificationManager =
                applicationContext.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            notificationManager.createNotificationChannel(channel)
            Log.d(TAG, "Notification channel created/updated.")
        }
    }

    private fun sendNotification(goal: Goal, notificationType: String) {
        val intent = Intent(applicationContext, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
            // You could add extras to navigate to a specific goal later
            putExtra("goal_id_notification", goal.id)
        }
        val pendingIntentFlags = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        } else {
            PendingIntent.FLAG_UPDATE_CURRENT
        }
        val pendingIntent = PendingIntent.getActivity(applicationContext, goal.id + notificationType.hashCode(), intent, pendingIntentFlags)


        val (title, content) = getNotificationContent(goal, notificationType)

        val notificationBuilder = NotificationCompat.Builder(applicationContext, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_launcher_foreground) // Replace with your app icon
            .setContentTitle(title)
            .setContentText(content)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setContentIntent(pendingIntent)
            .setAutoCancel(true) // Dismiss notification when tapped

        // Add actions based on notification type
        // e.g., Mark as done, Snooze, etc. (Future enhancement)

        with(NotificationManagerCompat.from(applicationContext)) {
            if (ActivityCompat.checkSelfPermission(
                    applicationContext,
                    Manifest.permission.POST_NOTIFICATIONS
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                Log.e(TAG, "POST_NOTIFICATIONS permission not granted. Cannot show notification.")
                // This should ideally be checked before scheduling or handled gracefully.
                // For Android 13+, you need to request this permission.
                return
            }
            // notificationId needs to be unique for each notification you want to show separately.
            // Using goal.id + type hash ensures pre, at, and post notifications for the same goal are distinct.
            val notificationId = goal.id + notificationType.hashCode()
            notify(notificationId, notificationBuilder.build())
            Log.i(TAG, "Notification sent for Goal ID: ${goal.id}, Type: $notificationType, Notification ID: $notificationId")
        }
    }

    private fun getNotificationContent(goal: Goal, notificationType: String): Pair<String, String> {
        val timeFormat = SimpleDateFormat("h:mm a", Locale.getDefault())
        val formattedTime = timeFormat.format(Date(goal.time))
        val behavior = goal.behavior
        val location = goal.location

        return when (notificationType) {
            NotificationScheduler.TYPE_PRE_GOAL -> {
                "Upcoming: $behavior" to "Reminder: You planned to '$behavior' at $formattedTime in '$location'."
            }
            NotificationScheduler.TYPE_AT_TIME -> {
                "It's time: $behavior!" to "Now is the time for '$behavior' at $formattedTime in '$location'."
            }
            NotificationScheduler.TYPE_POST_REVIEW -> {
                "Review: $behavior" to "How did '$behavior' at $formattedTime in '$location' go? Take a moment to review."
            }
            else -> "Intentive Goal" to "Check your goal: $behavior"
        }
    }
}