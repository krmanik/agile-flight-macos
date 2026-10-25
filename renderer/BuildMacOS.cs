// Batch-mode macOS build of the Flightmare Unity renderer.
// Copied into <flightmare_unity>/Assets/Editor/ by scripts/build_renderer.sh and run with
//   Unity -batchmode -quit -projectPath <project> -executeMethod BuildMacOS.Build -buildOutput <path.app>
using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Build.Reporting;

public static class BuildMacOS
{
    public static void Build()
    {
        string output = GetArg("-buildOutput") ?? "Build/RPG_Flightmare.app";
        // The public flightmare_unity repo omits the commercial environments
        // (Warehouse, Forest, Mountain, Wasteland); build whatever scenes exist.
        string[] enabled = EditorBuildSettings.scenes
            .Where(s => s.enabled)
            .Select(s => s.path)
            .ToArray();
        string[] scenes = enabled.Where(File.Exists).ToArray();
        foreach (string missing in enabled.Except(scenes))
            Console.WriteLine($"[BuildMacOS] Skipping missing scene: {missing}");
        // Empty racing arena (RaceArena.cs), scene_id 4 in sceneSchedule.cs
        RaceArena.Ensure();
        if (!scenes.Contains(RaceArena.ScenePath))
            scenes = scenes.Append(RaceArena.ScenePath).ToArray();

        EditorUserBuildSettings.SwitchActiveBuildTarget(BuildTargetGroup.Standalone, BuildTarget.StandaloneOSX);
        // Native Apple Silicon player (string API works across 2021.x editor versions)
        EditorUserBuildSettings.SetPlatformSettings("Standalone", "OSXUniversal", "Architecture", "ARM64");

        var options = new BuildPlayerOptions
        {
            scenes = scenes,
            locationPathName = output,
            target = BuildTarget.StandaloneOSX,
            options = BuildOptions.None,
        };

        BuildReport report = BuildPipeline.BuildPlayer(options);
        Console.WriteLine($"[BuildMacOS] {report.summary.result}: {output} " +
                          $"({report.summary.totalSize / (1024 * 1024)} MB, {report.summary.totalErrors} errors)");
        if (report.summary.result != BuildResult.Succeeded)
            EditorApplication.Exit(1);
    }

    static string GetArg(string name)
    {
        string[] args = Environment.GetCommandLineArgs();
        int i = Array.IndexOf(args, name);
        return i >= 0 && i + 1 < args.Length ? args[i + 1] : null;
    }
}
