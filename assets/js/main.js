/*
	Striped by HTML5 UP
	html5up.net | @ajlkn
	Free for personal and commercial use under the CCA 3.0 license (html5up.net/license)
*/

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

		// Height hack.
		/*
			var $sc = $('#sidebar, #content'), tid;

			$window
				.on('resize', function() {
					window.clearTimeout(tid);
					tid = window.setTimeout(function() {
						$sc.css('min-height', $document.height());
					}, 100);
				})
				.on('load', function() {
					$window.trigger('resize');
				})
				.trigger('resize');
		*/

		// Title Bar.
			var titleBar = $(
				'<div id="titleBar">' +
					'<a href="#sidebar" class="toggle"></a>' +
					'<span class="title">YangCazz</span>' +  // 默认标题
				'</div>'
			).appendTo($body);

			// 监听#logo元素加载
			var observer = new MutationObserver(function(mutations) {
				mutations.forEach(function(mutation) {
					if ($('#logo').length > 0) {
						titleBar.find('.title').text($('#logo').text());
						observer.disconnect();
					}
				});
			});

			// 开始观察
			observer.observe(document.body, {
				childList: true,
				subtree: true
			});

		// Sidebar
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
				});

})(jQuery);
