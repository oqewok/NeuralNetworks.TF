using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ImageConverter
{
	static class WidthHeidthRatioCalc
	{
		private static double _ratioSum;
		private static int _directoriesCount;

		public static double GetNewRatio(string rootPath)
		{
			Calculate(rootPath);

			var ratio = _ratioSum / _directoriesCount;

			File.WriteAllText(Path.Combine(rootPath, "average_size_ratio.txt"), ratio.ToString());
			
			return ratio;
		}

		public static double GetOldRatio(string rootPath)
		{
			var result = File.ReadAllText(Path.Combine(rootPath, "average_size_ratio.txt"));
			return double.Parse(result);
		}

		public static string FileExtension { get; } = ".jpeg";

		private static void Calculate(string path)
		{
			var directories = Directory.EnumerateDirectories(path);

			if(directories.Count() != 0)
			{
				foreach(var directory in directories)
				{
					Calculate(directory);
				}
			}
			else
			{
				var filesCount = Directory
					.EnumerateFiles(path)
					.Count(e => e.EndsWith(FileExtension));

				if(filesCount > 0)
				{
					var r = Ratio(path);
					_ratioSum += r;

					++_directoriesCount;
				}
			}
		}

		private static double Ratio(string directory)
		{
			var fullFileNames = Directory
				.EnumerateFiles(directory)
				.Where(e => e.EndsWith(FileExtension));

			var result = fullFileNames
				.Select(e => new Bitmap(e))
				.Average(s => (double)s.Width / s.Height);

			return result;
		}
	}
}
