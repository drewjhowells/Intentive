package com.example.intentive.ui.navigation

import androidx.compose.runtime.Composable
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.example.intentive.ui.screens.AddEditGoalScreen
import com.example.intentive.ui.screens.GoalListScreen
import com.example.intentive.ui.viewmodel.GoalViewModel

sealed class Screen(val route: String) {
    object IntentList : Screen("intentList")
    object AddIntent : Screen("addIntent")
    object EditIntent : Screen("editIntent/{goalId}") {
        fun createRoute(goalId: Int) = "editIntent/$goalId"
    }
}

@Composable
fun AppNavigation(goalViewModel: GoalViewModel = hiltViewModel()) {
    val navController = rememberNavController()

    NavHost(navController = navController, startDestination = Screen.IntentList.route) {
        composable(Screen.IntentList.route) {
            GoalListScreen(
                viewModel = goalViewModel,
                onAddGoalClicked = { navController.navigate(Screen.AddIntent.route) },
                onGoalClicked = { goalId ->
                    navController.navigate(Screen.EditIntent.createRoute(goalId))
                }
            )
        }
        composable(Screen.AddIntent.route) {
            AddEditGoalScreen(
                viewModel = goalViewModel,
                navController = navController,
                goalId = null // Indicates adding a new goal
            )
        }
        composable(
            route = Screen.EditIntent.route,
            arguments = listOf(navArgument("goalId") { type = NavType.IntType })
        ) { backStackEntry ->
            val goalId = backStackEntry.arguments?.getInt("goalId")
            AddEditGoalScreen(
                viewModel = goalViewModel,
                navController = navController,
                goalId = goalId
            )
        }
    }
}