package com.example.intentive.ui.screens

import android.annotation.SuppressLint
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.clickable
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.Notifications
import androidx.compose.material.icons.outlined.RadioButtonUnchecked
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.intentive.data.db.Goal
import com.example.intentive.ui.viewmodel.GoalViewModel
import java.text.SimpleDateFormat
import java.util.*

@OptIn(ExperimentalMaterial3Api::class, ExperimentalFoundationApi::class)
@SuppressLint("UnusedMaterial3ScaffoldPaddingParameter")
@Composable
fun GoalListScreen(
    viewModel: GoalViewModel,
    onAddGoalClicked: () -> Unit,
    onGoalClicked: (Int) -> Unit
) {
    val goals by viewModel.goals.collectAsState()

    Scaffold(
        floatingActionButton = {
            FloatingActionButton(onClick = onAddGoalClicked) {
                Icon(Icons.Filled.Add, "Add Intent")
            }
        },
        topBar = {
            TopAppBar(
                title = { 
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.Center
                    ) {
                        Text("Intentive")
                    }
                },
                actions = {
                    var showMenu by remember { mutableStateOf(false) }
                    
                    // Debug notification menu
                    Box {
                        IconButton(onClick = { showMenu = true }) {
                            Icon(Icons.Default.MoreVert, contentDescription = "Debug Menu")
                        }
                        
                        DropdownMenu(
                            expanded = showMenu,
                            onDismissRequest = { showMenu = false }
                        ) {
                            DropdownMenuItem(
                                text = { Text("Test 5s") },
                                onClick = { 
                                    viewModel.sendTestNotification()
                                    showMenu = false
                                }
                            )
                            DropdownMenuItem(
                                text = { Text("Test 3s") },
                                onClick = { 
                                    viewModel.sendImmediateTestNotification()
                                    showMenu = false
                                }
                            )
                            DropdownMenuItem(
                                text = { Text("Direct Test") },
                                onClick = { 
                                    viewModel.sendDirectTestNotification()
                                    showMenu = false
                                }
                            )
                            DropdownMenuItem(
                                text = { Text("Check Work") },
                                onClick = { 
                                    viewModel.checkScheduledWork()
                                    showMenu = false
                                }
                            )
                        }
                    }
                }
            )
        }
    ) { paddingValues ->
        if (goals.isEmpty()) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(paddingValues)
                    .padding(16.dp),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    "No intents yet. Tap the '+' button to add your first intention!",
                    style = MaterialTheme.typography.headlineSmall,
                    fontStyle = FontStyle.Italic
                )
            }
        } else {
            LazyColumn(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(paddingValues)
                    .padding(top = 8.dp) // Add some padding below the top bar
            ) {
                items(goals, key = { goal -> goal.id }) { goal ->
                    GoalItem(
                        goal = goal,
                        onGoalClicked = { onGoalClicked(goal.id) },
                        onToggleComplete = { viewModel.toggleGoalCompletion(goal) }
                    )
                    HorizontalDivider()
                }
            }
        }
    }
}

@Composable
fun GoalItem(goal: Goal, onGoalClicked: () -> Unit, onToggleComplete: () -> Unit) {
    val dateFormat = remember { SimpleDateFormat("EEE, MMM d 'at' h:mm a", Locale.getDefault()) }
    val timeString = remember(goal.time) { dateFormat.format(Date(goal.time)) }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 8.dp)
            .clickable { onGoalClicked() },
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Row(
            modifier = Modifier
                .padding(16.dp)
                .fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            IconButton(onClick = onToggleComplete) {
                Icon(
                    imageVector = if (goal.isCompleted) Icons.Filled.CheckCircle else Icons.Outlined.RadioButtonUnchecked,
                    contentDescription = if (goal.isCompleted) "Mark as Incomplete" else "Mark as Complete",
                    tint = if (goal.isCompleted) MaterialTheme.colorScheme.primary else Color.Gray
                )
            }
            Spacer(modifier = Modifier.width(12.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = goal.behavior,
                    fontWeight = FontWeight.Bold,
                    fontSize = 18.sp,
                    color = if (goal.isCompleted) Color.Gray else MaterialTheme.colorScheme.onSurface
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "At: ${goal.location}",
                    fontSize = 14.sp,
                    color = if (goal.isCompleted) Color.Gray else MaterialTheme.colorScheme.onSurfaceVariant
                )
                Text(
                    text = timeString,
                    fontSize = 14.sp,
                    color = if (goal.isCompleted) Color.Gray else MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}