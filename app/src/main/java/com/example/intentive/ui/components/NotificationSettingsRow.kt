package com.example.intentive.ui.components

import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Notifications
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.intentive.data.model.NotificationSettings
import com.example.intentive.data.model.NotificationTiming

@Composable
fun NotificationSettingsRow(
    settings: NotificationSettings,
    onSettingsChanged: (NotificationSettings) -> Unit,
    modifier: Modifier = Modifier
) {
    var showBeforeMenu by remember { mutableStateOf(false) }
    var showAfterMenu by remember { mutableStateOf(false) }

    Column(modifier = modifier) {
        Text(
            "Notifications:",
            style = MaterialTheme.typography.titleMedium,
            modifier = Modifier.padding(bottom = 8.dp)
        )
        
        // Time-based notification buttons
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            // Before button
            NotificationButton(
                text = "Before",
                subtitle = if (settings.beforeEnabled) settings.beforeTiming.displayName else null,
                isEnabled = settings.beforeEnabled,
                onClick = {
                    onSettingsChanged(settings.copy(beforeEnabled = !settings.beforeEnabled))
                },
                onLongPress = { showBeforeMenu = true },
                modifier = Modifier.weight(1f)
            )
            
            // At time button
            NotificationButton(
                text = "At Time",
                subtitle = null,
                isEnabled = settings.atTimeEnabled,
                onClick = {
                    onSettingsChanged(settings.copy(atTimeEnabled = !settings.atTimeEnabled))
                },
                onLongPress = null,
                modifier = Modifier.weight(1f)
            )
            
            // After button
            NotificationButton(
                text = "After",
                subtitle = if (settings.afterEnabled) settings.afterTiming.displayName else null,
                isEnabled = settings.afterEnabled,
                onClick = {
                    onSettingsChanged(settings.copy(afterEnabled = !settings.afterEnabled))
                },
                onLongPress = { showAfterMenu = true },
                modifier = Modifier.weight(1f)
            )
        }
    }

    // Before timing menu
    if (showBeforeMenu) {
        TimingSelectionDialog(
            title = "Notify Before",
            currentTiming = settings.beforeTiming,
            onTimingSelected = { timing ->
                onSettingsChanged(settings.copy(beforeTiming = timing, beforeEnabled = true))
                showBeforeMenu = false
            },
            onDismiss = { showBeforeMenu = false }
        )
    }

    // After timing menu
    if (showAfterMenu) {
        TimingSelectionDialog(
            title = "Notify After",
            currentTiming = settings.afterTiming,
            onTimingSelected = { timing ->
                onSettingsChanged(settings.copy(afterTiming = timing, afterEnabled = true))
                showAfterMenu = false
            },
            onDismiss = { showAfterMenu = false }
        )
    }
}

@Composable
fun LocationNotificationButton(
    isEnabled: Boolean,
    onToggle: () -> Unit,
    modifier: Modifier = Modifier
) {
    IconButton(
        onClick = onToggle,
        modifier = modifier
    ) {
        Icon(
            Icons.Default.Notifications,
            contentDescription = "Location notification",
            tint = if (isEnabled) MaterialTheme.colorScheme.primary else Color.Gray
        )
    }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun NotificationButton(
    text: String,
    subtitle: String?,
    isEnabled: Boolean,
    onClick: () -> Unit,
    onLongPress: (() -> Unit)?,
    modifier: Modifier = Modifier
) {
    val backgroundColor = if (isEnabled) {
        MaterialTheme.colorScheme.primaryContainer
    } else {
        MaterialTheme.colorScheme.surfaceVariant
    }
    
    val contentColor = if (isEnabled) {
        MaterialTheme.colorScheme.onPrimaryContainer
    } else {
        MaterialTheme.colorScheme.onSurfaceVariant
    }

    Card(
        modifier = modifier
            .combinedClickable(
                onClick = onClick,
                onLongClick = onLongPress
            ),
        colors = CardDefaults.cardColors(containerColor = backgroundColor)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = text,
                fontWeight = FontWeight.Medium,
                fontSize = 14.sp,
                color = contentColor
            )
            if (subtitle != null) {
                Text(
                    text = subtitle,
                    fontSize = 11.sp,
                    color = contentColor.copy(alpha = 0.7f),
                    modifier = Modifier.padding(top = 2.dp)
                )
            }
        }
    }
}

@Composable
private fun TimingSelectionDialog(
    title: String,
    currentTiming: NotificationTiming,
    onTimingSelected: (NotificationTiming) -> Unit,
    onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(title) },
        text = {
            Column {
                NotificationTiming.values().forEach { timing ->
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 4.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        RadioButton(
                            selected = timing == currentTiming,
                            onClick = { onTimingSelected(timing) }
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(timing.displayName)
                    }
                }
            }
        },
        confirmButton = {
            TextButton(onClick = onDismiss) {
                Text("Cancel")
            }
        }
    )
}
