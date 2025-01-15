(function($) {

	var	$window = $(window),
		$body = $('body'),
		$document = $(document);

	// Breakpoints.
		breakpoints({
			desktop:   [ '737px',   null     ],
			wide:      [ '1201px',  null     ],
			narrow:    [ '737px',   '1200px' ],
			narrower:  [ '737px',   '1000px' ],
			mobile:    [ null,      '736px'  ]
		});

	// Play initial animations on page load.
		$window.on('load', function() {
			window.setTimeout(function() {
				$body.removeClass('is-preload');
			}, 100);
		});

	// Nav.
		$('#sidebar')
			.panel({
				delay: 500,
				hideOnClick: true,
				hideOnSwipe: true,
				resetScroll: true,
				resetForms: true,
				side: 'left',
				target: $body,
				visibleClass: 'sidebar-visible'
			})
			.draggable({
				handle: '.sidebar-header',
				containment: 'window',
				cursor: 'move',
				start: function() {
					$('#sidebar').addClass('dragging');
				},
				stop: function() {
					$('#sidebar').removeClass('dragging');
				}
			});

	// Calendar click handler
	$('.calendar a').on('click', function(e) {
		e.preventDefault();
		var dateStr = $(this).attr('href').split('=')[1];
		loadArticleByDate(dateStr);
	});

	// Load article by date
	function loadArticleByDate(dateStr) {
		// Show loading state
		$('#loading').show();
		$('#article-container').addClass('loading');

		// Parse date string
		var dateParts = dateStr.split('-');
		var date = new Date(dateParts[0], dateParts[1]-1, dateParts[2]);
		
		// Get article data
		getArticleData(date)
			.then(function(article) {
				// Update main content
				$('#article-container').html(`
					<article class="box post post-excerpt">
						<header>
							<h2><a href="#">${article.title}</a></h2>
							<p>${article.subtitle}</p>
						</header>
						<div class="info">
							<span class="date">${formatDate(date)}</span>
						</div>
						${article.content}
					</article>
				`);
			})
			.catch(function(error) {
				// Show error message
				$('#article-container').html(`
					<article class="box post post-excerpt">
						<header>
							<h2><a href="#">加载失败</a></h2>
							<p>请稍后再试</p>
						</header>
						<div class="info">
							<span class="date">${formatDate(new Date())}</span>
						</div>
						<p>文章加载失败：${error.message}</p>
					</article>
				`);
			})
			.finally(function() {
				// Hide loading state
				$('#loading').hide();
				$('#article-container').removeClass('loading');
			});
	}

	// Format date to Chinese format
	function formatDate(date) {
		var year = date.getFullYear();
		var month = date.getMonth() + 1;
		var day = date.getDate();
		return year + '年' + month + '月' + day + '日';
	}

	// Get article data from JSON
	function getArticleData(date) {
		return $.ajax({
			url: 'assets/data/articles.json',
			dataType: 'json'
		}).then(function(data) {
			// Find article matching the date
			var article = data.articles.find(function(a) {
				return a.date === formatDateISO(date);
			});
			
			if (!article) {
				throw new Error('该日期没有文章');
			}
			
			return {
				title: article.title,
				subtitle: article.subtitle,
				content: article.content
			};
		});
	}

	// Format date to ISO format (YYYY-MM-DD)
	function formatDateISO(date) {
		var year = date.getFullYear();
		var month = ('0' + (date.getMonth() + 1)).slice(-2);
		var day = ('0' + date.getDate()).slice(-2);
		return year + '-' + month + '-' + day;
	}

})(jQuery);
