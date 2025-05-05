import gulp from 'gulp';
import sourcemaps from 'gulp-sourcemaps';
import * as sass from 'sass';
import gulpSass from 'gulp-sass';
import autoprefixer from 'gulp-autoprefixer';
import cleanCSS from 'gulp-clean-css';
import rename from 'gulp-rename';
import cssnano from 'gulp-cssnano';
import fileinclude from 'gulp-file-include';
import gulpIf from 'gulp-if';
import uglify from 'gulp-uglify';
import useref from 'gulp-useref';
import cached from 'gulp-cached';
import browserSync from 'browser-sync';
import { deleteAsync as del } from 'del';
import postcss from 'gulp-postcss';
import postcssImport from 'postcss-import';
const sassCompiler = gulpSass(sass);
import data from './package.json' assert { type: 'json' }

// BrowserSync instance
const browsersync = browserSync.create();

const isSourceMap = true;
const sourceMapWrite = isSourceMap ? "./" : false;

// BrowserSync Task
function browsersyncFn(done) {
  browsersync.init({
    server: {
      baseDir: ['./dist', './dist/html'],
    },
    port: 1112,
  });
  done();
}

function browsersyncReload(done) {
  browsersync.reload();
  done();
}

// Watch Task
function watchTask() {
  gulp.watch(['./src/assets/scss/**/*', '!./src/assets/switcher/*.scss'], gulp.series(scssTask, browsersyncReload));
  gulp.watch(['./src/assets/js/**/*'], gulp.series(jsTask, browsersyncReload));
  gulp.watch(['./src/html/**/*.html', './src/html/partials/*'], gulp.series(htmlTask, browsersyncReload));
  gulp.watch(['./src/assets/images/**/*'], gulp.series(copyImagesTask, browsersyncReload));
}

// HTML Task
function htmlTask() {
  const htmlFiles = './src/html/**/*.html';

  return gulp.src(htmlFiles)
    .pipe(fileinclude({
      prefix: '@SPK@',
      basepath: '@file',
      indent: true,
    }))
    .pipe(useref())
    .pipe(cached('html'))
    .pipe(gulpIf('*.js', uglify()))
    .pipe(gulpIf('*.css', cssnano({ svgo: false })))
    .pipe(gulp.dest('./dist/html'))
    .on('end', () => del('./dist/html/partials'));
}

// SCSS Task
function scssTask() {
  const scssFiles = './src/assets/scss/**/*.scss';
  const cssDest = './dist/assets/css';

  return gulp.src(scssFiles)
    .pipe(sourcemaps.init())
    .pipe(sassCompiler().on('error', sassCompiler.logError))
    .pipe(postcss([postcssImport()]))
    .pipe(autoprefixer())
    .pipe(gulp.dest(cssDest))
    .pipe(sourcemaps.init())
    .pipe(sassCompiler().on('error', sassCompiler.logError))
    .pipe(postcss([postcssImport()]))
    .pipe(autoprefixer())
    .pipe(cleanCSS())
    .pipe(rename({ suffix: '.min' }))
    .pipe(sourcemaps.write(sourceMapWrite))
    .pipe(gulp.dest(cssDest));
}

// JavaScript Task
function jsTask() {
  const jsFilePath = './dist/assets/js';

  return gulp.src('./src/assets/js/**/*')
    .pipe(sourcemaps.init())
    // .pipe(gulpIf(file => file.path.endsWith('.js'), uglify().on('error', e => console.error(e)))) // enable to get minified files
    // .pipe(rename({ suffix: '.min' })) // enable to get .min extension
    // .pipe(sourcemaps.write(sourceMapWrite)) // enable to get .map files
    .pipe(gulp.dest(jsFilePath));
}

let myData = []
let linWas = './node_modules/**/**/*'
Object.keys(data.dependencies).map((ele)=>{
  myData.push(linWas.replace('**',ele))
})

function npmdist() {
  return [
    ...myData,
  ];
}

function copyLibsTask() {
  const destPath = 'dist/assets/libs';

  return gulp.src(npmdist(), { base: './node_modules', encoding: false })
    .pipe(rename(path => {
      path.dirname = path.dirname.replace(/\/dist/, '').replace(/\\dist/, '');
    }))
    .pipe(gulp.dest(destPath));
}

function copyImagesTask() {
  const imagePath = './dist/assets/images';

  return gulp.src('./src/assets/images/**/*', {encoding: false})
    .pipe(gulp.dest(imagePath));
}

// Copy All Task
function copyAllTask() {
  const assetsPath = './dist/assets';

  return gulp.src([
    './src/assets/**/*',
    '!./src/assets/**/',
    '!./src/assets/js/**/*',
    '!./src/assets/images/**/*', // Exclude images to avoid duplication
  ], {encoding: false})
  .pipe(gulp.dest(assetsPath));
}

// Clean Dist Task
function cleanDistTask() {
  return del('./dist');
}

// Build Task
const build = gulp.series(
  cleanDistTask,
  gulp.parallel(copyAllTask, copyImagesTask, copyLibsTask),
  htmlTask,
  scssTask,
  jsTask
);

// Default Task
const defaults = gulp.series(build, gulp.parallel(browsersyncFn, watchTask));

// Export tasks
export {
  browsersyncFn,
  browsersyncReload,
  jsTask as js,
  scssTask as scss,
  htmlTask as html,
  cleanDistTask as cleanDist,
  copyAllTask as copyAll,
  copyImagesTask,
  watchTask as watch,
  build,
  defaults as default
};
 